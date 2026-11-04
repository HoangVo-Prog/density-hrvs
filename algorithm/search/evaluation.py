"""Verifier and detector primitives shared by region and candidate evaluation."""

import json
from pathlib import Path
import time

from numpy.typing import NDArray
from PIL import Image

from ..core.contracts import MatchAnswer, SAM3Detection, SearchNode, VerifierScore
from ..core.diagnostics import stage_index
from ..core.errors import InvalidScoreResponse
from ..core.imaging import to_pil_rgb
from ..models.scoring import ScoringView
from .runtime import RunState
from .recovery import recover_empty_region


def score_crop(
    state: RunState,
    crop: Image.Image,
    description: str,
    view_type: ScoringView,
) -> VerifierScore:
    """Score one exact crop, reusing an identical crop/query assessment."""
    key = state.view_key(crop, view_type, description)
    if key in state.score_cache:
        state.usage["cache_hits"] += 1
        state.on_event(
            "score_cache_hit",
            f"N{state.context.node_id}: reuse an identical crop/query assessment.",
            {"node_id": state.context.node_id, "usage": state.usage.copy()},
        )
        return state.score_cache[key]
    state.usage["logical_score_requests"] = state.usage.get("logical_score_requests", 0) + 1
    tick = time.perf_counter()
    try:
        if state.custom_scorer is None:
            if state.local_scorer is None:
                raise RuntimeError("Local scorer was not initialized")
            result = state.local_scorer(
                crop,
                description,
                view_type=view_type,
                on_request=state.charge_local,
            )
        else:
            state.charge_request()
            result = state.custom_scorer(crop, description)
    finally:
        state.record_cost("score", time.perf_counter() - tick)
    state.score_cache[key] = result
    return result


def placeholder_rating(
    reason: str,
    answer: MatchAnswer = "No",
    score: float = 0.0,
) -> VerifierScore:
    """Create an explicit not-run assessment instead of inventing a zero score."""
    return {
        "score": score,
        "answer": answer,
        "reason": reason,
        "model_id": "not-run",
    }


def candidate_query_text(state: RunState) -> str:
    """Choose the full query or detector phrase for candidate evidence sheets."""
    if state.settings.get("detector_prompt_mode") == "objects_and_references":
        return state.question
    if state.settings.get("detector_prompt_mode") == "references":
        refs = state.settings["reference_plan"]["references"]
        return ("Verify a REFERENCE object used to locate the requested target. "
                "The candidate box is a reference, NOT the requested target. "
                "Check its known attributes and location against these reference descriptions: "
                + json.dumps(refs) + ". Do not require the box to enclose the final target. "
                "Do not mistake a similar object elsewhere for the required reference. "
                "Original question: " + state.question)
    plan = state.settings.get("query_plan") if state.settings.get("structured_query", False) else None
    if plan and plan.get("attribute_region"):
        return (f"Identify the owning/reference entity: {plan['entity']['concept']}. "
                f"The requested {plan['attribute_region']['concept']} will be inspected separately. "
                "At this stage verify the owner's identity and the required spatial/reference relations. "
                "A whole-owner rectangle is intentional; do not reject it merely for enclosing more than the part. "
                f"Original question: {state.question}")
    if state.settings.get("candidate_query_mode", "full") == "target_phrase":
        return state.settings["sam_prompt"] or state.query
    return state.query


def _save_invalid_region(
    state: RunState,
    node: SearchNode,
    folder: Path | None,
    exc: InvalidScoreResponse,
) -> None:
    if folder is None:
        return
    (folder / "invalid_score_response.json").write_text(
        json.dumps(
            {
                "query": state.query,
                "model": state.settings.get("scoring_backend", ""),
                "raw_responses": exc.responses,
                "error": str(exc),
            },
            indent=2,
        )
    )
    stage_index(folder)


def assess_region(
    state: RunState,
    node: SearchNode,
    pixels: NDArray,
    folder: Path | None,
) -> VerifierScore:
    """Legacy whole-region assessment retained for A/B comparisons."""
    state.on_event(
        "qwen_started",
        f"N{node['node_id']}: assess the whole unmasked region, not individual boxes.",
        {
            "node_id": node["node_id"],
            "bbox": node["bbox"],
            "image_size": (pixels.shape[1], pixels.shape[0]),
        },
    )
    try:
        assessment = score_crop(state, to_pil_rgb(pixels), state.query, "region")
    except InvalidScoreResponse as exc:
        _save_invalid_region(state, node, folder, exc)
        raise
    if folder is not None:
        (folder / "region_score.json").write_text(json.dumps(assessment, indent=2))
    state.on_event(
        "qwen_completed",
        f"N{node['node_id']}: Qwen relevance {assessment['score']:.3f}, {assessment['answer']}.",
        {
            "node_id": node["node_id"],
            "qwen": assessment,
            "usage": state.usage.copy(),
        },
    )
    return assessment


def detect_region(
    state: RunState,
    node: SearchNode,
    pixels: NDArray,
) -> list[SAM3Detection]:
    """Run and cache confidence-filtered SAM3 proposals for this region."""
    if node["node_id"] not in state.detection_cache:
        tick = time.perf_counter()
        state.on_event(
            "sam3_started",
            f"N{node['node_id']}: detect concept {state.settings['sam_prompt'] or state.query!r}.",
            {
                "node_id": node["node_id"],
                "detector_prompt": state.settings["sam_prompt"] or state.query,
                "threshold": state.settings["sam3_threshold"],
                "proposal_limit": state.settings["max_proposals_per_node"],
            },
        )
        if state.settings.get("detector_prompt_mode") in {"references", "objects_and_references"}:
            detections = []
            plan=state.settings["reference_plan"]
            for reference in plan.get("objects",plan["references"]):
                state.usage["primary_detector_calls"] = state.usage.get("primary_detector_calls", 0) + 1
                role=reference.get("role","reference")
                counter="target_detector_calls" if role=="target" else "reference_detector_calls"
                state.usage[counter]=state.usage.get(counter,0)+1
                found = state.detector(pixels, reference["prompt"])
                for detection in found:
                    detection = {**detection, "reference_id": reference["id"],
                                 "prompt": reference["prompt"], "proposal_role": role}
                    detections.append(detection)
                    x, y, _, _ = node["bbox"]
                    a, b, c, d = detection["bbox"]
                    state.settings["reference_detections"].append({
                        "node_id": node["node_id"], "reference_id": reference["id"],
                        "prompt": reference["prompt"], "bbox": [x+a, y+b, x+c, y+d],
                        "sam3_score": detection["sam3_score"], "proposal_role": role,
                        "sam3_selection": detection.get("sam3_selection", "above_threshold")})
                if any(d.get("sam3_selection") == "top_k_fallback" for d in found):
                    state.usage["sam3_top_k_fallback_calls"] = state.usage.get("sam3_top_k_fallback_calls", 0) + 1
        else:
            state.usage["primary_detector_calls"] = state.usage.get("primary_detector_calls", 0) + 1
            detections = state.detector(pixels, state.settings["sam_prompt"] or state.query)
        if not detections:
            detections = recover_empty_region(state, node, pixels)
        state.detection_cache[node["node_id"]] = sorted(
            [
                detection
                for detection in detections
                if detection["sam3_score"] is None
                or detection["sam3_score"] > state.settings["sam3_threshold"]
                or (state.settings.get("sam3_fallback_top_k", 0) > 0
                    and detection.get("sam3_selection") == "top_k_fallback")
            ],
            key=lambda detection: detection["sam3_score"] if detection["sam3_score"] is not None else detection.get("detector_score", 0.0),
            reverse=True,
        )
        state.record_cost("detect", time.perf_counter() - tick)
    return state.detection_cache[node["node_id"]]
