"""Node evaluation orchestration: split, legacy regions, candidates, and ranking."""

import json
from pathlib import Path
import time

from numpy.typing import NDArray

from ..core.contracts import (
    BoundingBox,
    CandidateCheck,
    ChildRegion,
    NodeEvidence,
    RankedRegion,
    RegionScoreComponents,
    SAM3Detection,
    SearchNode,
    VerifierScore,
)
from ..core.diagnostics import stage_index
from ..core.geometry import box_iou
from ..core.imaging import to_pil_rgb
from ..reporting.artifacts import save_node_detections, save_node_input
from .candidates import verify_candidates, verify_region_candidates
from .evaluation import assess_region, detect_region, placeholder_rating
from .partitioning import semantic_children
from .region_scoring import (
    aggregate_verified_region,
    candidate_score,
    distinct_detections,
    score_region_boxes,
)
from .runtime import RunState


def split_node(state: RunState, node: SearchNode) -> list[ChildRegion]:
    """Partition a node once, prepending discovered seeds at the root."""
    if node["node_id"] not in state.split_cache:
        tick = time.perf_counter()
        semantic = semantic_children(
            state.rgb,
            node,
            state.settings,
            state.feature_extractor,
            state.node_folder(node["node_id"]),
            state.on_event,
        )
        if node["node_id"] == 0 and state.seed_children:
            children = list(state.seed_children)
            children.extend(
                child
                for child in semantic
                if all(
                    box_iou(child["bbox"], seed["bbox"]) <= 0.8
                    for seed in state.seed_children
                )
            )
            state.split_cache[node["node_id"]] = children
        else:
            state.split_cache[node["node_id"]] = semantic
        state.record_cost("split", time.perf_counter() - tick)
    return state.split_cache[node["node_id"]]


def _proposal_identity(
    state: RunState, proposal: RankedRegion
) -> tuple[BoundingBox, str]:
    return (
        proposal["bbox"],
        state.view_key(to_pil_rgb(proposal["image"]), "proposal", state.query),
    )


def register_proposals(state: RunState, proposals: list[RankedRegion]) -> None:
    """Deduplicate observed proposals, keeping the highest-scoring copy."""
    for proposal in proposals:
        identity = _proposal_identity(state, proposal)
        if identity not in state.seen_proposals:
            state.seen_proposals[identity] = len(state.observed_regions)
            state.observed_regions.append(proposal)
        elif (
            proposal["score"]
            > state.observed_regions[state.seen_proposals[identity]]["score"]
        ):
            state.observed_regions[state.seen_proposals[identity]] = proposal


def _record_cluster(
    state: RunState,
    node: SearchNode,
    pixels: NDArray,
    score: float,
    assessment: VerifierScore,
    components: RegionScoreComponents,
) -> None:
    region_ids = node.get("region_ids", [])
    state.observed_clusters.append(
        {
            "cluster_id": node["node_id"],
            "region_ids": region_ids,
            "n_regions": len(region_ids),
            "pixel_count": int(node["mask"].sum()),
            "image": pixels,
            "bbox": node["bbox"],
            "score": score,
            "verifier": assessment,
            "scoring_method": "qwen3vl_local_visual_query_match",
            "score_components": components,
        }
    )


def _legacy_proposals(
    state: RunState,
    node: SearchNode,
    unique: list[SAM3Detection],
    region_score: float,
    assessment: VerifierScore,
    components: RegionScoreComponents,
) -> list[RankedRegion]:
    x0, y0, _, _ = node["bbox"]
    proposals: list[RankedRegion] = []
    for detection in unique:
        a, b, c, d = detection["bbox"]
        proposals.append(
            {
                **detection,
                "cluster_id": node["node_id"],
                "cluster_score": region_score,
                "local_bbox": detection["bbox"],
                "bbox": (x0 + a, y0 + b, x0 + c, y0 + d),
                "score": region_score,
                "verifier": assessment,
                "scoring_scope": "region",
                "score_components": components,
                "scoring_method": "qwen3vl_local_visual_query_match",
            }
        )
    return proposals


def evaluate_legacy(
    state: RunState,
    node: SearchNode,
    pixels: NDArray,
    folder: Path | None,
) -> NodeEvidence:
    """Score the whole region, then rank its detector boxes."""
    assessment = assess_region(state, node, pixels, folder)
    detections = detect_region(state, node, pixels)
    components, unique = score_region_boxes(
        detections, assessment["score"], state.settings
    )
    region_score = components["combined_score"]
    state.on_event(
        "region_scored",
        f"N{node['node_id']}: reward {region_score:.4f}; {len(detections)} boxes before NMS, {len(unique)} after NMS.",
        {
            "node_id": node["node_id"],
            "score_components": components,
            "boxes_before_nms": len(detections),
            "boxes_after_nms": len(unique),
            "nms_iou": state.settings.get("region_nms_iou", 0.7),
            "legacy_region_qwen": True,
        },
    )
    _record_cluster(state, node, pixels, region_score, assessment, components)
    save_node_detections(pixels, detections, unique, components, folder)
    proposals = _legacy_proposals(
        state, node, unique, region_score, assessment, components
    )
    register_proposals(state, proposals)
    if folder is not None:
        stage_index(folder)
    return {
        "assessment": assessment,
        "proposals": proposals,
        "raw_reward": region_score,
    }


def _best_assessment(
    checks: list[CandidateCheck],
    distinct: list[SAM3Detection],
) -> VerifierScore:
    best = max(checks, key=lambda item: item["candidate_score"], default=None)
    if best is not None and "qwen" in best:
        return best["qwen"]
    if not distinct:
        return placeholder_rating("No SAM3 candidates; Qwen was not called.")
    if not checks:
        return placeholder_rating("No candidate was verified; Qwen was not called.")
    return placeholder_rating("All checked candidates were rejected by Qwen.")


def _candidate_proposals(
    state: RunState,
    node: SearchNode,
    distinct: list[SAM3Detection],
    checks: list[CandidateCheck],
    components: RegionScoreComponents,
) -> list[RankedRegion]:
    checks_by_index = {check["detection_index"]: check for check in checks}
    x0, y0, _, _ = node["bbox"]
    accept = state.settings.get("candidate_accept_score", 0.5)
    proposals: list[RankedRegion] = []
    for detection_index, detection in enumerate(distinct):
        a, b, c, d = detection["bbox"]
        check = checks_by_index.get(detection_index)
        rating = (
            check["qwen"]
            if check is not None and "qwen" in check
            else placeholder_rating(
                "Not checked: candidate verification limit reached.",
                answer="Uncertain",
            )
        )
        qwen_value = (
            rating["score"]
            if rating["answer"] != "No" and rating["model_id"] != "not-run"
            else None
        )
        fused = candidate_score(qwen_value, detection["sam3_score"], state.settings)
        verified = bool(
            rating["model_id"] != "not-run"
            and rating["answer"] == "Yes"
            and rating["score"] >= accept
        )
        proposal: RankedRegion = {
            **detection,
            "cluster_id": node["node_id"],
            "cluster_score": components["combined_score"],
            "local_bbox": detection["bbox"],
            "bbox": (x0 + a, y0 + b, x0 + c, y0 + d),
            "score": fused,
            "verifier": rating,
            "scoring_scope": "proposal",
            "candidate_rank": detection_index + 1,
            "candidate_score": fused,
            "candidate_verified": verified,
            "score_components": components,
            "scoring_method": "qwen3vl_local_visual_query_match",
        }
        if check is not None and "qwen" in check:
            proposal["candidate_qwen"] = check["qwen"]
        if check is not None and "evidence" in check:
            proposal["evidence"] = check["evidence"]
        proposals.append(proposal)
    return proposals


def evaluate_candidates(
    state: RunState,
    node: SearchNode,
    pixels: NDArray,
    folder: Path | None,
) -> NodeEvidence:
    """Detect candidates first, then verify their context/detail sheets."""
    detections = detect_region(state, node, pixels)
    distinct = distinct_detections(detections, state.settings)
    if not distinct:
        state.usage["gated_regions"] = state.usage.get("gated_regions", 0) + 1
        state.on_event(
            "region_gated",
            f"N{node['node_id']}: no SAM3 candidates; Qwen not called and region reward is zero.",
            {
                "node_id": node["node_id"],
                "gate": "no_candidates",
                "boxes_before_nms": len(detections),
            },
        )
    if distinct:
        if state.settings.get("verification_mode", "region") == "region":
            checks = verify_region_candidates(state, node, distinct, pixels, folder)
            if checks is None:
                checks = verify_candidates(state, node, distinct, folder)
        else:
            checks = verify_candidates(state, node, distinct, folder)
    else:
        checks = []
    state.checked_cache[node["node_id"]] = checks
    components = aggregate_verified_region(distinct, checks, state.settings)
    components["raw_box_count"] = len(detections)
    components["unique_box_count"] = len(distinct)
    assessment = _best_assessment(checks, distinct)
    region_score = components["combined_score"]
    if folder is not None:
        (folder / "region_score.json").write_text(json.dumps(assessment, indent=2))
    state.on_event(
        "region_scored",
        f"N{node['node_id']}: reward {region_score:.4f}; gate {components.get('gate')}; {len(detections)} boxes before NMS, {len(distinct)} after NMS, {len(checks)} checked, {components.get('verified_box_count', 0)} verified.",
        {
            "node_id": node["node_id"],
            "score_components": components,
            "boxes_before_nms": len(detections),
            "boxes_after_nms": len(distinct),
            "nms_iou": state.settings.get("region_nms_iou", 0.7),
            "candidate_checks": len(checks),
            "verified_box_count": components.get("verified_box_count", 0),
            "gate": components.get("gate"),
        },
    )
    _record_cluster(state, node, pixels, region_score, assessment, components)
    save_node_detections(
        pixels,
        detections,
        distinct,
        components,
        folder,
        checks=checks,
    )
    proposals = _candidate_proposals(state, node, distinct, checks, components)
    register_proposals(state, proposals)
    if folder is not None:
        stage_index(folder)
    return {
        "assessment": assessment,
        "proposals": proposals,
        "raw_reward": region_score,
        "candidate_checks": checks,
    }


def evaluate_node(state: RunState, node: SearchNode) -> NodeEvidence:
    """Evaluate one node on the candidate-first or legacy region path."""
    state.context.node_id = node["node_id"]
    x0, y0, x_end, y_end = node["bbox"]
    pixels = state.rgb[y0:y_end, x0:x_end].copy()
    folder = state.node_folder(node["node_id"])
    state.context.folder = folder
    save_node_input(state.rgb, node, pixels, folder)
    if state.settings.get("sam3_first", True):
        return evaluate_candidates(state, node, pixels, folder)
    return evaluate_legacy(state, node, pixels, folder)


def estimate_node(state: RunState, node: SearchNode) -> float:
    """Forecast remaining seconds from observed costs and populated caches."""
    x1, y1, x2, y2 = node["bbox"]
    pixels = state.rgb[y1:y2, x1:x2]
    pending = state.detection_cache.get(node["node_id"])
    detection_cost = 0.0 if pending is not None else state.mean_cost("detect", 1.0)
    if state.settings.get("sam3_first", True):
        if state.settings.get("verification_mode", "region") == "region":
            can_check = state.settings.get("region_candidate_limit", 5) > 0
            checks = (
                1
                if can_check
                and node["node_id"] not in state.checked_cache
                and pending != []
                else 0
            )
        else:
            limit = state.settings.get("candidate_check_limit", 2)
            if pending is None:
                checks = max(1, min(limit, 2))
            elif not pending:
                checks = 0
            elif node["node_id"] in state.checked_cache:
                checks = 0
            else:
                checks = min(limit, len(pending))
        score_cost = checks * state.mean_cost("score", 3.0)
    else:
        score_cost = (
            0.0
            if state.view_key(to_pil_rgb(pixels), "region", state.query)
            in state.score_cache
            else state.mean_cost("score", 3.0)
        )
    split_cost = (
        state.mean_cost("split", 1.0)
        if node["depth"] < state.settings["max_depth"]
        and node["node_id"] not in state.split_cache
        else 0.0
    )
    return score_cost + detection_cost + split_cost
