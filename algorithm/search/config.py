"""Validated defaults for the bounded recursive visual search."""

import math

from ..core.contracts import SearchConfig


def default_search_config() -> SearchConfig:
    """Return independent, editable defaults; 0 HTTP calls/proposals means unlimited."""
    return {
        "strategy": "agent-uct",
        "device": "auto",
        "max_api_calls": 32,
        "max_iterations": 100,
        "max_depth": 3,
        "branching_factor": 5,
        "min_region_pixels": 256,
        "max_proposals_per_node": 16,
        "exploration_weight": 1.4,
        "cost_weight": 0.2,
        "reference_cost_seconds": 10.0,
        "sam3_threshold": 0.1,
        "mask_threshold": 0.5,
        "sam_prompt": "",
        "rewrite_sam_prompt": True,
        "sam3_first": True,
        "verification_mode": "candidate",
        "discovery_mode": "off",
        "discovery_tiles": 3,
        "discovery_candidates": 5,
        "discovery_seeds": 3,
        "discovery_overlap": 0.10,
        "region_candidate_limit": 5,
        "region_fallback_limit": 3,
        "candidate_check_limit": 2,
        "candidate_accept_score": 0.5,
        "candidate_qwen_weight": 0.7,
        "candidate_sam3_weight": 0.3,
        "candidate_query_mode": "full",
        "feature_max_side": 1024,
        "segmentation_max_side": 1024,
        "verification_max_side": 1536,
        "context_padding": 384,
        "detail_padding": 16,
        "n_segments_values": (200, 400, 600, 800, 1000, 1500),
        "entropy_bins": 8,
        "slic": {
            "n_segments": 200,
            "compactness": 4.0,
            "sigma": 0.0,
            "max_num_iter": 30,
            "enforce_connectivity": True,
            "convert2lab": True,
            "slic_zero": False,
        },
        "stop_on_match": False,
        "match_threshold": 0.9,
        "max_local_scores": 32,
        "cluster_spatial_weight": 0.75,
        "min_cluster_occupancy": 0.25,
        "max_cluster_aspect": 6.0,
        "min_largest_component": 0.7,
        "proposal_backend": "facebook/sam3:unmasked-region",
        "scoring_backend": "Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203",
        "region_qwen_weight": 0.65,
        "region_confidence_weight": 0.2,
        "region_count_weight": 0.1,
        "region_diversity_weight": 0.05,
        "region_box_count_cap": 5,
        "region_nms_iou": 0.7,
    }


def _validate_region_weights(config: SearchConfig) -> None:
    weights = [
        config.get("region_qwen_weight", 0.65),
        config.get("region_confidence_weight", 0.2),
        config.get("region_count_weight", 0.1),
        config.get("region_diversity_weight", 0.05),
    ]
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(
            "Region score weights must be finite, nonnegative and sum to a positive value"
        )
    count_cap = config.get("region_box_count_cap", 5)
    if type(count_cap) is not int or count_cap < 1:
        raise ValueError("region_box_count_cap must be a positive integer")
    nms_iou = config.get("region_nms_iou", 0.7)
    if not math.isfinite(nms_iou) or not 0 <= nms_iou <= 1:
        raise ValueError("region_nms_iou must be in [0, 1]")


def _validate_candidate_settings(config: SearchConfig) -> None:
    if not isinstance(config.get("rewrite_sam_prompt", True), bool):
        raise ValueError("rewrite_sam_prompt must be a boolean")
    if not isinstance(config.get("sam3_first", True), bool):
        raise ValueError("sam3_first must be a boolean")
    weights = [
        config.get("candidate_qwen_weight", 0.7),
        config.get("candidate_sam3_weight", 0.3),
    ]
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(
            "Candidate score weights must be finite, nonnegative and sum to a positive value"
        )
    check_limit = config.get("candidate_check_limit", 2)
    if type(check_limit) is not int or check_limit < 0:
        raise ValueError("candidate_check_limit must be a nonnegative integer")
    if config.get("verification_mode", "region") not in {"region", "candidate"}:
        raise ValueError("verification_mode must be 'region' or 'candidate'")
    region_limit = config.get("region_candidate_limit", 5)
    if type(region_limit) is not int or region_limit < 0:
        raise ValueError("region_candidate_limit must be a nonnegative integer")
    fallback_limit = config.get("region_fallback_limit", 3)
    if type(fallback_limit) is not int or fallback_limit < 0:
        raise ValueError("region_fallback_limit must be a nonnegative integer")
    if config.get("discovery_mode", "off") not in {"off", "sam3_qwen"}:
        raise ValueError("discovery_mode must be 'off' or 'sam3_qwen'")
    for name, minimum in (
        ("discovery_tiles", 1),
        ("discovery_candidates", 0),
        ("discovery_seeds", 0),
    ):
        value = config.get(name, 0)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    overlap = config.get("discovery_overlap", 0.10)
    if not math.isfinite(overlap) or not 0 <= overlap < 0.5:
        raise ValueError("discovery_overlap must be in [0, 0.5)")
    accept_score = config.get("candidate_accept_score", 0.5)
    if not math.isfinite(accept_score) or not 0 <= accept_score <= 1:
        raise ValueError("candidate_accept_score must be in [0, 1]")
    if config.get("candidate_query_mode", "full") not in {"full", "target_phrase"}:
        raise ValueError("candidate_query_mode must be 'full' or 'target_phrase'")


def _validate_budgets(config: SearchConfig) -> None:
    for name in ("max_api_calls", "max_proposals_per_node"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    for name in (
        "max_iterations",
        "max_depth",
        "branching_factor",
        "min_region_pixels",
        "verification_max_side",
        "segmentation_max_side",
    ):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")


def _validate_image_scales(config: SearchConfig) -> None:
    if config["feature_max_side"] < 32:
        raise ValueError("feature_max_side must be at least 32 pixels.")
    if config["verification_max_side"] < 512:
        raise ValueError(
            "verification_max_side must be at least 512 for context/detail views."
        )
    if config["context_padding"] < 0 or config["detail_padding"] < 0:
        raise ValueError("Context and detail padding must be nonnegative.")


def _validate_allocation(config: SearchConfig) -> None:
    for name in ("exploration_weight", "cost_weight"):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if (
        not math.isfinite(config["reference_cost_seconds"])
        or config["reference_cost_seconds"] <= 0
    ):
        raise ValueError("reference_cost_seconds must be finite and positive.")


def _validate_thresholds(config: SearchConfig) -> None:
    for name in ("sam3_threshold", "mask_threshold", "match_threshold"):
        if not math.isfinite(config[name]) or not 0 <= config[name] <= 1:
            raise ValueError(f"{name} must be between zero and one.")


def _validate_strategy(config: SearchConfig) -> None:
    if config["strategy"] not in {"agent-uct", "uct", "greedy"}:
        raise ValueError("Unknown search strategy.")
    if not config["n_segments_values"] or any(
        type(n) is not int or n < 1 for n in config["n_segments_values"]
    ):
        raise ValueError("Provide positive SLIC candidate counts.")
    if config["entropy_bins"] < 2:
        raise ValueError("entropy_bins must be at least two.")


def validate_search_config(config: SearchConfig) -> None:
    """Reject invalid budgets/scales before local model loading or API charges."""
    _validate_region_weights(config)
    _validate_candidate_settings(config)
    _validate_budgets(config)
    _validate_image_scales(config)
    _validate_allocation(config)
    _validate_thresholds(config)
    _validate_strategy(config)
