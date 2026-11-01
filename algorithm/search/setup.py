"""Validate requests and lazily assemble the concrete model callables."""

from collections.abc import Callable
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray

from ..core.contracts import SAM3Detection, SearchConfig, SearchUsage
from ..core.events import EventSink, SAM3CandidateRecord
from ..core.imaging import to_pil_rgb
from ..models.qwen_scoring import LocalScorer, load_qwen_scorer
from ..models.regions import exec_dinov3_pca_on_image, load_dinov3
from ..models.sam3 import exec_sam3_on_image, load_sam3_model
from .config import default_search_config, validate_search_config
from .runtime import (
    Detector,
    FeatureExtractor,
    NodeContext,
    RunState,
    SearchRequest,
)


def _resolve_sam_prompt(
    settings: SearchConfig,
    query: str,
    scorer: LocalScorer,
    usage: SearchUsage,
    on_event: EventSink,
) -> None:
    """Ask Qwen to choose the SAM3 localization target from the full query.

    The resolver receives the question-derived search description directly. It
    is not anchored to the initial noun parsed by ``question_target``, so it can
    turn "the shoes of the man..." into the person/upper body that the ground
    truth actually localizes.
    """
    if not settings.get("rewrite_sam_prompt", True):
        return
    resolver = getattr(scorer, "resolve", None)
    rewriter = getattr(scorer, "rewrite", None)
    if not callable(resolver) and not callable(rewriter):
        return

    original_prompt = settings["sam_prompt"] or query
    started = time.perf_counter()
    usage["local_prompt_calls"] = usage.get("local_prompt_calls", 0) + 1
    failure = ""
    source = "target_resolver"
    try:
        if callable(resolver):
            resolved = resolver(query)
        else:
            source = "rewrite_fallback"
            resolved = rewriter(query, original_prompt)
    except Exception as exc:
        resolved = ""
        failure = f"{type(exc).__name__}: {exc}"
    if not isinstance(resolved, str):
        resolved = ""
        failure = "resolver returned a non-string"
    resolved = resolved.strip()
    if not resolved:
        failure = failure or "empty resolution"
    if resolved:
        settings["sam_prompt"] = resolved
        on_event(
            "sam_prompt_rewritten",
            f"SAM3 target resolved: {original_prompt!r} -> {resolved!r}.",
            {
                "original_prompt": original_prompt,
                "rewritten_prompt": resolved,
                "duration_seconds": time.perf_counter() - started,
                "model_id": settings.get("scoring_backend", "Qwen3-VL-2B"),
                "resolver": source,
            },
        )
    else:
        on_event(
            "sam_prompt_rewrite_failed",
            "Keeping the initial SAM3 prompt after Qwen target resolution failed.",
            {
                "original_prompt": original_prompt,
                "reason": failure or "empty resolution",
            },
        )


def _prepare_local_scorer(
    settings: SearchConfig,
    query: str,
    provided: LocalScorer | None,
    usage: SearchUsage,
    on_event: EventSink,
) -> LocalScorer:
    loading = time.perf_counter()
    on_event(
        "model_loading",
        "Prepare local Qwen verifier.",
        {"model_id": settings.get("scoring_backend", "Qwen3-VL-2B")},
    )
    scorer = provided if provided is not None else load_qwen_scorer(settings["device"])
    on_event(
        "model_ready",
        "Local Qwen verifier ready.",
        {
            "model_id": settings.get("scoring_backend", "Qwen3-VL-2B"),
            "duration_seconds": time.perf_counter() - loading,
        },
    )
    _resolve_sam_prompt(settings, query, scorer, usage, on_event)
    return scorer


def _load_feature_extractor(
    settings: SearchConfig,
    progress: Callable[[str], None],
    on_event: EventSink,
) -> FeatureExtractor:
    progress("Loading pinned DINOv3 checkpoint.")
    on_event(
        "model_loading",
        "Load pinned local DINOv3 weights.",
        {"model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
    )
    model, processor = load_dinov3(device=settings["device"])
    extractor = partial(
        exec_dinov3_pca_on_image,
        model=model,
        processor=processor,
        max_side=settings["feature_max_side"],
    )
    on_event(
        "model_ready",
        "DINOv3 feature extractor ready.",
        {"model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
    )
    return extractor


def _load_detector(
    settings: SearchConfig,
    context: NodeContext,
    progress: Callable[[str], None],
    on_event: EventSink,
) -> Detector:
    progress("Loading local SAM3; Qwen will verify candidate evidence sheets.")
    on_event(
        "model_loading",
        "Load pinned local SAM3 weights.",
        {"model_id": "facebook/sam3"},
    )
    model, processor = load_sam3_model(settings["device"])
    on_event("model_ready", "SAM3 detector ready.", {"model_id": "facebook/sam3"})

    def save_candidates(records: list[SAM3CandidateRecord]) -> None:
        if context.folder is not None:
            (context.folder / "sam3_all_candidates.json").write_text(
                json.dumps(records, indent=2)
            )
        on_event(
            "sam3_filtering",
            f"N{context.node_id}: threshold/cap retained {sum(r['retained'] for r in records)} of {len(records)} native boxes before NMS.",
            {
                "node_id": context.node_id,
                "threshold": settings["sam3_threshold"],
                "proposal_limit": settings["max_proposals_per_node"],
                "sam_candidates": records,
            },
        )

    def detector(crop: NDArray, concept: str) -> list[SAM3Detection]:
        return exec_sam3_on_image(
            crop,
            model,
            processor,
            prompt=concept,
            threshold=settings["sam3_threshold"],
            mask_threshold=settings["mask_threshold"],
            max_detections=settings["max_proposals_per_node"],
            on_candidates=save_candidates,
        )

    return detector


def prepare_run(request: SearchRequest) -> RunState:
    """Validate a request, load omitted models, and return mutable run state.

    Args:
        request: Typed search inputs, including optional injected models.

    Returns:
        A state object with the original image, resolved config, usage ledger,
        model callables, and per-run caches.

    Raises:
        ValueError: The query or config is invalid.
    """
    started = time.perf_counter()
    settings = default_search_config() if request.config is None else request.config
    validate_search_config(settings)
    if not request.query.strip():
        raise ValueError("Provide a nonempty query.")
    source_image = to_pil_rgb(request.image)
    rgb = np.asarray(source_image)
    artifacts = Path(request.artifact_dir) if request.artifact_dir is not None else None
    use_sam3_first = settings.get("sam3_first", True)
    request.on_event(
        "search_started",
        "Start local SAM3 candidate search with context/detail Qwen verification.",
        {
            "query": request.query,
            "detector_prompt": settings["sam_prompt"] or request.query,
            "config": settings,
            "image_size": (rgb.shape[1], rgb.shape[0]),
            "sam3_first": use_sam3_first,
        },
    )
    usage: SearchUsage = {
        "api_calls": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reported_cost_usd": 0.0,
        "candidate_checks": 0,
        "gated_regions": 0,
    }
    context = NodeContext()
    custom_scorer = request.scorer
    local_scorer = (
        _prepare_local_scorer(
            settings,
            request.query,
            request.local_scorer,
            usage,
            request.on_event,
        )
        if custom_scorer is None
        else None
    )
    feature_extractor = (
        request.feature_extractor
        if request.feature_extractor is not None
        else _load_feature_extractor(settings, request.progress, request.on_event)
    )
    detector = (
        request.detector
        if request.detector is not None
        else _load_detector(
            settings,
            context,
            request.progress,
            request.on_event,
        )
    )
    return RunState(
        query=request.query,
        question=(request.question or request.query).strip(),
        settings=settings,
        usage=usage,
        source_image=source_image,
        rgb=rgb,
        artifacts=artifacts,
        on_event=request.on_event,
        progress=request.progress,
        custom_scorer=custom_scorer,
        local_scorer=local_scorer,
        feature_extractor=feature_extractor,
        detector=detector,
        context=context,
        started_at=started,
    )
