"""Verify SAM3 boxes on evidence sheets or together on the whole cluster."""

from dataclasses import dataclass
import json
from pathlib import Path
import time

from numpy.typing import NDArray
from PIL import Image

from ..core.contracts import (
    BoundingBox,
    CandidateCheck,
    CandidatePrompt,
    ProposalEvidence,
    SAM3Detection,
    SearchNode,
)
from ..core.diagnostics import stage_index
from ..core.errors import InvalidScoreResponse
from ..core.imaging import clamp_global_box, to_pil_rgb
from ..models.evidence import (
    build_proposal_evidence,
    build_region_candidate_sheet,
    evidence_info,
)
from ..models.scoring import REGION_TIE_MARGIN
from .evaluation import candidate_query_text, score_crop
from .region_scoring import candidate_score
from .runtime import RunState


@dataclass(frozen=True)
class CandidateEvidence:
    """One candidate's exact verifier sheet and its image-coordinate box."""

    evidence: ProposalEvidence
    image: Image.Image
    global_box: BoundingBox


def _build_candidate_evidence(
    state: RunState,
    node: SearchNode,
    detection: SAM3Detection,
    rank: int,
    folder: Path | None,
) -> CandidateEvidence | None:
    local_box = detection["bbox"]
    global_box = clamp_global_box(
        node["bbox"],
        local_box,
        (state.rgb.shape[1], state.rgb.shape[0]),
    )
    if global_box is None:
        return None
    try:
        evidence = build_proposal_evidence(
            state.source_image,
            global_box,
            max_side=state.settings.get("verification_max_side", 1536),
            context_padding=state.settings["context_padding"],
            detail_padding=state.settings["detail_padding"],
        )
    except ValueError as exc:
        state.on_event(
            "candidate_verification_failed",
            f"N{node['node_id']} candidate {rank}: cannot build evidence ({exc}).",
            {
                "node_id": node["node_id"],
                "candidate_rank": rank,
                "bbox": global_box,
                "error": str(exc),
            },
        )
        return None
    image = Image.fromarray(evidence["image"])
    if folder is not None:
        image.save(folder / f"candidate_{rank:03d}_evidence.png")
        (folder / f"candidate_{rank:03d}_input.json").write_text(
            json.dumps(
                {
                    "candidate_rank": rank,
                    "local_bbox": local_box,
                    "bbox": global_box,
                    "evidence": evidence_info(evidence),
                },
                indent=2,
            )
        )
    return CandidateEvidence(evidence=evidence, image=image, global_box=global_box)


def _save_invalid_candidate(
    state: RunState,
    candidate: CandidateEvidence,
    rank: int,
    folder: Path | None,
    exc: InvalidScoreResponse,
) -> None:
    if folder is None:
        return
    (folder / f"invalid_candidate_{rank:03d}.json").write_text(
        json.dumps(
            {
                "query": candidate_query_text(state),
                "candidate_rank": rank,
                "bbox": candidate.global_box,
                "raw_responses": exc.responses,
                "error": str(exc),
            },
            indent=2,
        )
    )
    stage_index(folder)


def verify_candidate(
    state: RunState,
    node: SearchNode,
    detection: SAM3Detection,
    rank: int,
    folder: Path | None,
) -> CandidateCheck | None:
    """Verify one SAM3 box on a context/detail sheet."""
    candidate = _build_candidate_evidence(state, node, detection, rank, folder)
    if candidate is None:
        return None
    state.on_event(
        "qwen_started",
        f"N{node['node_id']} candidate {rank}: verify the boxed target on a context/detail sheet.",
        {
            "node_id": node["node_id"],
            "candidate_rank": rank,
            "bbox": candidate.global_box,
            "view_type": "context_detail",
            "image_size": candidate.image.size,
        },
    )
    state.on_event(
        "candidate_verification_started",
        f"N{node['node_id']}: verify SAM3 candidate {rank} ({detection['sam3_score'] or 0:.3f}).",
        {
            "node_id": node["node_id"],
            "candidate_rank": rank,
            "bbox": candidate.global_box,
            "sam3_score": detection["sam3_score"],
            "view_type": "context_detail",
        },
    )
    try:
        rating = score_crop(
            state,
            candidate.image,
            candidate_query_text(state),
            "context_detail",
        )
    except InvalidScoreResponse as exc:
        _save_invalid_candidate(state, candidate, rank, folder, exc)
        raise
    if folder is not None:
        (folder / f"candidate_{rank:03d}_score.json").write_text(
            json.dumps(rating, indent=2)
        )
    fused = candidate_score(
        rating["score"] if rating["answer"] != "No" else None,
        detection["sam3_score"],
        state.settings,
    )
    check: CandidateCheck = {
        "rank": rank,
        "detection_index": rank - 1,
        "local_bbox": detection["bbox"],
        "bbox": candidate.global_box,
        "sam3_score": detection["sam3_score"],
        "checked": True,
        "candidate_score": fused,
        "qwen": rating,
        "evidence": candidate.evidence,
    }
    state.on_event(
        "qwen_completed",
        f"N{node['node_id']} candidate {rank}: Qwen {rating['score']:.3f}, {rating['answer']}.",
        {
            "node_id": node["node_id"],
            "candidate_rank": rank,
            "qwen": rating,
            "usage": state.usage.copy(),
        },
    )
    state.on_event(
        "candidate_verification_completed",
        f"N{node['node_id']} candidate {rank}: Qwen {rating['score']:.3f}, {rating['answer']}; fused {fused:.3f}.",
        {
            "node_id": node["node_id"],
            "candidate_rank": rank,
            "bbox": candidate.global_box,
            "sam3_score": detection["sam3_score"],
            "qwen": rating,
            "candidate_score": fused,
        },
    )
    return check


def verify_candidates(
    state: RunState,
    node: SearchNode,
    distinct: list[SAM3Detection],
    folder: Path | None,
) -> list[CandidateCheck]:
    """Verify candidates in confidence order, stopping at the first clear Yes."""
    limit = state.settings.get("candidate_check_limit", 2)
    accept = state.settings.get("candidate_accept_score", 0.5)
    checks: list[CandidateCheck] = []
    for index, detection in enumerate(distinct[:limit]):
        rank = index + 1
        check = verify_candidate(state, node, detection, rank, folder)
        if check is None:
            continue
        checks.append(check)
        if check["qwen"]["answer"] == "Yes" and check["qwen"]["score"] >= accept:
            break
    state.usage["candidate_checks"] = state.usage.get("candidate_checks", 0) + len(
        checks
    )
    return checks


def verify_selected_candidates(
    state: RunState,
    node: SearchNode,
    distinct: list[SAM3Detection],
    indices: list[int],
    folder: Path | None,
) -> list[CandidateCheck]:
    """Verify selected detections with per-candidate context/detail sheets."""
    checks: list[CandidateCheck] = []
    for index in sorted(set(indices)):
        if not 0 <= index < len(distinct):
            continue
        check = verify_candidate(state, node, distinct[index], index + 1, folder)
        if check is not None:
            checks.append(check)
    return checks


def verify_region_candidates(
    state: RunState,
    node: SearchNode,
    distinct: list[SAM3Detection],
    pixels: NDArray,
    folder: Path | None,
) -> list[CandidateCheck] | None:
    """Rank the top SAM3 boxes on the whole cluster, then detail-check ambiguity.

    Returns None when the local scorer does not implement ``score_candidates``,
    so the caller can fall back to per-candidate context/detail evidence sheets.
    """
    scorer = state.local_scorer
    verifier = getattr(scorer, "score_candidates", None)
    if not callable(verifier):
        return None
    limit = state.settings.get("region_candidate_limit", 5)
    if limit <= 0 or not distinct:
        return []

    subset = distinct[:limit]
    fallback_limit = state.settings.get("region_fallback_limit", 3)
    label = state.settings["sam_prompt"] or state.query
    candidates: list[CandidatePrompt] = [
        {
            "index": index + 1,
            "label": label,
            "confidence": detection["sam3_score"],
            "bbox": detection["bbox"],
        }
        for index, detection in enumerate(subset)
    ]
    sheet = build_region_candidate_sheet(
        to_pil_rgb(pixels),
        candidates,
        max_side=state.settings.get("verification_max_side", 1536),
    )
    if folder is not None:
        sheet.save(folder / "region_candidates.png")
        (folder / "region_candidates.json").write_text(
            json.dumps(candidates, indent=2)
        )
    state.on_event(
        "qwen_started",
        f"N{node['node_id']}: rank {len(candidates)} SAM3 boxes on the whole cluster region.",
        {
            "node_id": node["node_id"],
            "candidate_indices": [candidate["index"] for candidate in candidates],
            "view_type": "region_overview",
            "image_size": sheet.size,
        },
    )
    tick = time.perf_counter()
    try:
        assessment = verifier(
            sheet,
            state.query,
            candidates,
            on_request=state.charge_local,
        )
    except InvalidScoreResponse as exc:
        if folder is not None:
            (folder / "invalid_region_candidates.json").write_text(
                json.dumps(
                    {
                        "query": state.query,
                        "raw_responses": exc.responses,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
            stage_index(folder)
        fallback_indices = list(range(min(fallback_limit, len(distinct))))
        fallback_checks = verify_selected_candidates(
            state, node, distinct, fallback_indices, folder
        )
        state.usage["candidate_checks"] = state.usage.get(
            "candidate_checks", 0
        ) + len(fallback_checks)
        state.on_event(
            "qwen_completed",
            f"N{node['node_id']}: invalid region JSON; detail-checked {len(fallback_checks)} candidates instead.",
            {
                "node_id": node["node_id"],
                "fallback": "invalid_json",
                "candidate_indices": [index + 1 for index in fallback_indices],
                "raw_responses": exc.responses,
                "usage": state.usage.copy(),
            },
        )
        return fallback_checks
    finally:
        state.record_cost("score", time.perf_counter() - tick)

    region_checks: dict[int, CandidateCheck] = {}
    for index, detection in enumerate(subset):
        rank = index + 1
        rating = assessment["ratings"].get(rank)
        if rating is None:
            continue
        global_box = clamp_global_box(
            node["bbox"],
            detection["bbox"],
            (state.rgb.shape[1], state.rgb.shape[0]),
        )
        if global_box is None:
            continue
        fused = candidate_score(
            rating["score"] if rating["answer"] != "No" else None,
            detection["sam3_score"],
            state.settings,
        )
        region_checks[index] = {
            "rank": rank,
            "detection_index": index,
            "local_bbox": detection["bbox"],
            "bbox": global_box,
            "sam3_score": detection["sam3_score"],
            "checked": True,
            "candidate_score": fused,
            "qwen": rating,
        }

    if folder is not None:
        (folder / "region_candidate_scores.json").write_text(
            json.dumps(assessment, indent=2)
        )

    fallback_indices: list[int] = []
    if assessment["ambiguous"] and fallback_limit > 0:
        best = assessment["ratings"][assessment["best_index"]]
        if best["answer"] == "Yes":
            selected = [
                index
                for index in assessment["ranking"]
                if assessment["ratings"][index]["score"]
                >= best["score"] - REGION_TIE_MARGIN
            ]
            if len(selected) < 2:
                selected = assessment["ranking"][:fallback_limit]
        else:
            selected = assessment["ranking"][:fallback_limit]
        fallback_indices = [index - 1 for index in selected[: max(limit, 1)]]

    merged = dict(region_checks)
    if fallback_indices:
        for check in verify_selected_candidates(
            state, node, distinct, fallback_indices, folder
        ):
            merged[check["detection_index"]] = check
    checks = [merged[index] for index in sorted(merged)]
    state.usage["candidate_checks"] = state.usage.get("candidate_checks", 0) + len(
        checks
    )
    state.on_event(
        "qwen_completed",
        f"N{node['node_id']}: region ranking checked {len(checks)} boxes; fallback {len(fallback_indices)}.",
        {
            "node_id": node["node_id"],
            "assessment": assessment,
            "fallback_indices": [index + 1 for index in fallback_indices],
            "usage": state.usage.copy(),
        },
    )
    return checks
