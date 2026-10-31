"""SAM3 box statistics and Qwen relevance combined into an auditable region reward."""

from itertools import combinations
import math

from ..core.contracts import (
    BoundingBox,
    CandidateCheck,
    RegionScoreComponents,
    SAM3Detection,
    SearchConfig,
)


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Pairwise xyxy intersection over union, with exclusive maxima."""
    a, b, c, d = first
    e, f, g, h = second
    intersection = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    union = (c - a) * (d - b) + (g - e) * (h - f) - intersection
    return intersection / union if union > 0 else 0.0


def distinct_detections(
    detections: list[SAM3Detection], config: SearchConfig
) -> list[SAM3Detection]:
    """Return confidence-ordered boxes after duplicate suppression."""
    distinct: list[SAM3Detection] = []
    for detection in sorted(
        detections, key=lambda d: d["sam3_score"] or 0.0, reverse=True
    ):
        confidence = detection["sam3_score"]
        if (
            confidence is None
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Candidate statistics require valid SAM3 confidence")
        if all(
            bbox_iou(detection["bbox"], kept["bbox"])
            <= config.get("region_nms_iou", 0.7)
            for kept in distinct
        ):
            distinct.append(detection)
    return distinct


def candidate_score(
    qwen_score: float | None,
    sam3_score: float | None,
    config: SearchConfig,
) -> float:
    """Fuse one candidate's verifier rating with its detector confidence."""
    qwen = 0.0 if qwen_score is None else qwen_score
    detector = 0.0 if sam3_score is None else sam3_score
    for value, label in ((qwen, "Qwen"), (detector, "SAM3")):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{label} candidate score must be finite and in [0, 1]")
    weights = (
        config.get("candidate_qwen_weight", 0.7),
        config.get("candidate_sam3_weight", 0.3),
    )
    total = sum(weights)
    if total <= 0:
        raise ValueError("Candidate score weights must sum to a positive value")
    return (weights[0] * qwen + weights[1] * detector) / total


def aggregate_verified_region(
    detections: list[SAM3Detection],
    checks: list[CandidateCheck],
    config: SearchConfig,
) -> RegionScoreComponents:
    """Reward only candidates that Qwen accepted on their evidence sheet."""
    best_qwen = max(
        (
            check["qwen"]["score"]
            for check in checks
            if "qwen" in check and check["qwen"]["answer"] != "No"
        ),
        default=0.0,
    )
    verified = [
        check
        for check in checks
        if "qwen" in check
        and check["qwen"]["answer"] == "Yes"
        and check["qwen"]["score"] >= config.get("candidate_accept_score", 0.5)
    ]
    verified_boxes = [
        detections[check["detection_index"]]
        for check in verified
        if 0 <= check["detection_index"] < len(detections)
    ]
    confidence = (
        sum(box["sam3_score"] or 0.0 for box in verified_boxes) / len(verified_boxes)
        if verified_boxes
        else 0.0
    )
    pairs = [
        bbox_iou(first["bbox"], second["bbox"])
        for first, second in combinations(verified_boxes, 2)
    ]
    overlap = sum(pairs) / len(pairs) if pairs else 0.0
    cap = config.get("region_box_count_cap", 5)
    count = min(len(verified_boxes), cap) / cap
    diversity = 1.0 - overlap if verified_boxes else 0.0
    weights = (
        config.get("region_qwen_weight", 0.65),
        config.get("region_confidence_weight", 0.2),
        config.get("region_count_weight", 0.1),
        config.get("region_diversity_weight", 0.05),
    )
    combined = (
        sum(
            weight * value
            for weight, value in zip(weights, (best_qwen, confidence, count, diversity))
        )
        / sum(weights)
        if verified_boxes
        else 0.0
    )
    if not detections:
        gate = "no_candidates"
    elif not checks:
        gate = "budget"
    elif not verified_boxes:
        gate = "unverified"
    else:
        gate = "verified"
    return {
        "qwen_score": best_qwen,
        "raw_box_count": len(detections),
        "unique_box_count": len(detections),
        "mean_confidence": confidence,
        "count_score": count,
        "mean_pairwise_iou": overlap,
        "diversity_score": diversity,
        "combined_score": combined,
        "gate": gate,
        "verified_box_count": len(verified_boxes),
        "candidate_checks": len(checks),
    }


def score_region_boxes(
    detections: list[SAM3Detection], qwen_score: float, config: SearchConfig
) -> tuple[RegionScoreComponents, list[SAM3Detection]]:
    """Legacy whole-region reward retained for A/B comparisons.

    Args:
        detections: Above-threshold SAM3 boxes in a single region's coordinates.
        qwen_score: Whole-region query relevance, not per-box verification.
        config: Nonnegative weights, NMS IoU and positive count saturation cap.

    Returns:
        Complete score breakdown and confidence-ordered distinct proposals.
        With no boxes, detector terms are zero; Qwen can still guide exploration.
    """
    if not math.isfinite(qwen_score) or not 0 <= qwen_score <= 1:
        raise ValueError("Qwen region score must be finite and in [0, 1]")
    distinct = distinct_detections(detections, config)
    confidence = (
        sum(d["sam3_score"] or 0.0 for d in distinct) / len(distinct)
        if distinct
        else 0.0
    )
    pairs = [bbox_iou(a["bbox"], b["bbox"]) for a, b in combinations(distinct, 2)]
    overlap = sum(pairs) / len(pairs) if pairs else 0.0
    count = min(len(distinct), config.get("region_box_count_cap", 5)) / config.get(
        "region_box_count_cap", 5
    )
    diversity = 1.0 - overlap if distinct else 0.0
    weights = (
        config.get("region_qwen_weight", 0.65),
        config.get("region_confidence_weight", 0.2),
        config.get("region_count_weight", 0.1),
        config.get("region_diversity_weight", 0.05),
    )
    combined = sum(
        weight * value
        for weight, value in zip(weights, (qwen_score, confidence, count, diversity))
    ) / sum(weights)
    return {
        "qwen_score": qwen_score,
        "raw_box_count": len(detections),
        "unique_box_count": len(distinct),
        "mean_confidence": confidence,
        "count_score": count,
        "mean_pairwise_iou": overlap,
        "diversity_score": diversity,
        "combined_score": combined,
    }, distinct
