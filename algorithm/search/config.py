"""Validated defaults for the bounded recursive visual search."""

import math

from ..core.contracts import SearchConfig


def default_search_config() -> SearchConfig:
    """Return independent, editable defaults; 0 HTTP calls/proposals means unlimited."""
    return {
        "strategy": "agent-uct",
        "partition_mode": "semantic",
        "spatial_overlap": .1,
        "device": "auto",
        "qwen_model": "Qwen/Qwen3-VL-2B-Instruct",
        "qwen_revision": None,
        "qwen_cache_dir": None,
        "qwen_quantization": "none",
        "model_residency": "resident",
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
        "sam3_fallback_top_k": 0,
        "mask_threshold": 0.5,
        "sam_prompt": "",
        "rewrite_sam_prompt": True,
        "sam3_first": True,
        "verification_mode": "candidate",
        "question_region_score_source": "yes_no_token_probability",
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
        "answer_question": True,
        "require_verified_answer": True,
        "strict_fallback_limit": True,
        "answer_context_padding": 384,
        "answer_view": "legacy_crop",
        "grounding_dino_fallback": False,
        "max_fallback_detector_calls": 1,
        "fallback_cache_dir": None,
        "fallback_threshold": .25,
        "fallback_text_threshold": .25,
        "fallback_proposal_limit": 8,
        "recovery_promising_score": .7,
        "comparative_reranking": False,
        "rerank_candidate_limit": 5,
        "structured_query": False,
        "part_refinement": False,
        "part_crop_retry": False,
        "finalization_reserve": 4,
        "max_part_detector_calls": 1,
        "part_candidate_limit": 3,
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
    if config.get("verification_mode", "region") not in {"region", "candidate", "question_region"}:
        raise ValueError("verification_mode must be region, candidate or question_region")
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
    for name, default in (("merge_small_fraction", .02), ("merge_max_bbox_fraction", .65)):
        value = config.get(name, default)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    overlap = config.get("discovery_overlap", 0.10)
    if not math.isfinite(overlap) or not 0 <= overlap < 0.5:
        raise ValueError("discovery_overlap must be in [0, 0.5)")
    accept_score = config.get("candidate_accept_score", 0.5)
    if not math.isfinite(accept_score) or not 0 <= accept_score <= 1:
        raise ValueError("candidate_accept_score must be in [0, 1]")
    if config.get("candidate_query_mode", "full") not in {"full", "target_phrase"}:
        raise ValueError("candidate_query_mode must be 'full' or 'target_phrase'")
    if not isinstance(config.get("answer_question", True), bool):
        raise ValueError("answer_question must be a boolean")
    for flag in ("require_verified_answer", "strict_fallback_limit", "structured_query", "part_refinement", "comparative_reranking", "grounding_dino_fallback", "part_crop_retry"):
        if not isinstance(config.get(flag, True), bool):
            raise ValueError(f"{flag} must be a boolean")
    if config.get("answer_view", "legacy_crop") not in {"marked_detail", "legacy_crop", "spatial_context"}:
        raise ValueError("answer_view must be marked_detail, legacy_crop or spatial_context")
    if config.get("region_spatial_context",False):
        if config.get("verification_mode")!="question_region":
            raise ValueError("Spatial region context requires question_region verification")
        side=config.get("spatial_overview_max_side",768)
        if type(side) is not int or not 128<=side<=1536:
            raise ValueError("spatial_overview_max_side must be an integer from 128 through 1536")
    for name in ("finalization_reserve", "max_part_detector_calls", "part_candidate_limit", "max_fallback_detector_calls"):
        value = config.get(name, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if config.get("part_refinement", False) and not config.get("structured_query", False):
        raise ValueError("part_refinement requires structured_query")
    if type(config.get("rerank_candidate_limit", 5)) is not int or not 1 <= config.get("rerank_candidate_limit", 5) <= 5:
        raise ValueError("rerank_candidate_limit must be an integer in [1, 5]")
    for name in ("fallback_threshold", "fallback_text_threshold", "recovery_promising_score"):
        value = config.get(name, .25)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if type(config.get("fallback_proposal_limit", 8)) is not int or config.get("fallback_proposal_limit", 8) < 1:
        raise ValueError("fallback_proposal_limit must be a positive integer")
    if config.get("part_crop_retry", False) and (
        not config.get("part_refinement", False) or config.get("max_part_detector_calls", 1) < 2):
        raise ValueError("part_crop_retry requires part_refinement and max_part_detector_calls >= 2")
    answer_padding = config.get("answer_context_padding", 384)
    if type(answer_padding) is not int or answer_padding < 0:
        raise ValueError("answer_context_padding must be a nonnegative integer")


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
    top_k = config.get("sam3_fallback_top_k", 0)
    if type(top_k) is not int or top_k < 0 or top_k > 200:
        raise ValueError("sam3_fallback_top_k must be an integer from 0 through 200")
    if top_k and config["max_proposals_per_node"] and config["max_proposals_per_node"] < top_k:
        raise ValueError("max_proposals_per_node must accommodate sam3_fallback_top_k")
    for name in ("sam3_threshold", "mask_threshold", "match_threshold"):
        if not math.isfinite(config[name]) or not 0 <= config[name] <= 1:
            raise ValueError(f"{name} must be between zero and one.")


def _validate_strategy(config: SearchConfig) -> None:
    if config.get("primary_detector", "sam3") not in {"sam3", "llmdet"}:
        raise ValueError("Unknown primary detector")
    if config.get("detector_prompt_mode", "category") not in {"category", "locating_statement", "references", "objects_and_references"}:
        raise ValueError("Unknown detector prompt mode")
    if config.get("detector_prompt_mode") == "objects_and_references":
        if config.get("verification_mode") != "question_region" or config.get("primary_detector","sam3") != "sam3":
            raise ValueError("Object/reference mode requires SAM3 and question_region verification")
        if config.get("part_refinement") or config.get("grounding_dino_fallback") or config.get("discovery_mode","off") != "off":
            raise ValueError("Disable alternate target detector paths for question_region mode")
    if config.get("detector_prompt_mode") == "references":
        if config.get("primary_detector", "sam3") != "sam3":
            raise ValueError("Reference mode requires SAM3")
        if config.get("part_refinement") or config.get("grounding_dino_fallback") or config.get("discovery_mode", "off") != "off":
            raise ValueError("Reference mode requires target/part detector paths and legacy discovery off")
    if config.get("primary_detector") == "llmdet":
        if config.get("part_refinement"):
            raise ValueError("Direct LLMDet target statements require separate part refinement off")
        threshold = config.get("llmdet_threshold", .25)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid LLMDet threshold")
    if config["strategy"] not in {"agent-uct", "uct", "greedy", "bfs"}:
        raise ValueError("Unknown search strategy.")
    if config.get("partition_mode", "semantic") not in {"semantic", "spatial", "adjacent_merge"}:
        raise ValueError("partition_mode must be semantic, spatial or adjacent_merge")
    overlap = config.get("spatial_overlap", .1)
    if not math.isfinite(overlap) or not 0 <= overlap < .5:
        raise ValueError("spatial_overlap must be in [0, .5)")
    if not config["n_segments_values"] or any(
        type(n) is not int or n < 1 for n in config["n_segments_values"]
    ):
        raise ValueError("Provide positive SLIC candidate counts.")
    if config["entropy_bins"] < 2:
        raise ValueError("entropy_bins must be at least two.")


def validate_search_config(config: SearchConfig) -> None:
    """Reject invalid budgets/scales before local model loading or API charges."""
    if config.get("model_residency", "resident") not in {"resident", "staged"}:
        raise ValueError("model_residency must be resident or staged")
    if config.get("qwen_quantization", "none") not in {"none", "nf4"}:
        raise ValueError("qwen_quantization must be none or nf4")
    _validate_region_weights(config)
    _validate_candidate_settings(config)
    _validate_budgets(config)
    _validate_image_scales(config)
    _validate_allocation(config)
    _validate_thresholds(config)
    _validate_strategy(config)
