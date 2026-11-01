"""Thin search orchestration: prepare state, run the tree, finalize artifacts."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..core.contracts import (
    ChildRegion,
    SAM3Detection,
    SearchConfig,
    SearchResult,
    VerifierScore,
)
from ..core.diagnostics import save_tree, stage_index
from ..core.events import EventSink, ignore_event
from ..models.qwen_scoring import LocalScorer
from ..reporting.artifacts import save_search_result
from .answering import answer_top_region
from .discovery import discover_seed_regions
from .pipeline import estimate_node, evaluate_node, split_node
from .runtime import RunState, SearchRequest
from .setup import prepare_run
from .tree import explore_tree

__all__ = ["run_visual_search", "save_search_result"]


FeatureExtractor = Callable[[NDArray], NDArray[np.float32]]
Detector = Callable[[NDArray, str], list[SAM3Detection]]
Scorer = Callable[[Image.Image, str], VerifierScore]

TREE_INDEX_HTML = (
    "<html><h1>Algorithm stages</h1>"
    '<a href="tree/index.html">Tree search and per-node stage galleries</a></html>'
)


def run_visual_search(
    image: NDArray,
    query: str,
    *,
    question: str | None = None,
    config: SearchConfig | None = None,
    scorer: Scorer | None = None,
    detector: Detector | None = None,
    feature_extractor: FeatureExtractor | None = None,
    progress: Callable[[str], None] = print,
    local_scorer: LocalScorer | None = None,
    artifact_dir: str | Path | None = None,
    on_event: EventSink = ignore_event,
) -> SearchResult:
    """Run the complete bounded recursive algorithm with per-run caches.

    Args:
        image: Original uint8 RGB image, or float RGB in [0, 1].
        query: Complete target description, including all required attributes.
        question: Original question for the final-answer stage; defaults to query.
        config: Editable default_search_config() controls; omitted uses defaults.
        scorer: Legacy custom scorer. Each invocation counts as one API unit;
            production scoring uses local_scorer and its separate compute ledger.
        detector: Optional crop detector; omitted loads local SAM3.
        feature_extractor: Optional DINO PCA encoder; omitted loads pinned DINOv3.
        progress: Human-readable status callback with no credentials/payloads.
        local_scorer: Shared pinned local Qwen callable; loaded when omitted.
        artifact_dir: Directory for per-node DINO/SLIC/clustering/model outputs
            and every tree-allocation snapshot; None disables these diagnostics.
        on_event: Structured observer for diagnostics and progress UIs.

    Returns:
        Ranked evidence, complete tree statistics, physical API usage and stopping
        reason. Partially collected evidence is retained if a budget interrupts a
        node; that node contributes no backed-up return until fully evaluated.
    """
    state = prepare_run(
        SearchRequest(
            image=image,
            query=query,
            question=question,
            config=config,
            scorer=scorer,
            detector=detector,
            feature_extractor=feature_extractor,
            progress=progress,
            local_scorer=local_scorer,
            artifact_dir=artifact_dir,
            on_event=on_event,
        )
    )
    state.seed_children = discover_seed_regions(state)
    observer = (
        (lambda nodes, trace: save_tree(state.artifacts / "tree", nodes, trace))
        if state.artifacts is not None
        else None
    )
    result = explore_tree(
        _root_region(state),
        state.query,
        state.settings,
        partial(evaluate_node, state),
        partial(split_node, state),
        partial(estimate_node, state),
        state.usage,
        state.progress,
        on_step=observer,
        on_event=state.on_event,
    )
    _attach_rankings(state, result)
    _attach_final_answer(state, result)
    result["elapsed_seconds"] = time.perf_counter() - state.started_at
    _finalize_artifacts(state, result)
    state.on_event(
        "search_finished",
        f"Search stopped: {result['stop_reason']}; {len(result['ranked_regions'])} retained boxes.",
        {
            "stop_reason": result["stop_reason"],
            "usage": state.usage.copy(),
            "error": result["error"],
            "duration_seconds": result["elapsed_seconds"],
        },
    )
    return result


def _root_region(state: RunState) -> ChildRegion:
    """Return the whole-image region that seeds the search tree."""
    height, width = state.rgb.shape[:2]
    return {
        "bbox": (0, 0, width, height),
        "mask": np.ones((height, width), dtype=bool),
    }


def _attach_rankings(state: RunState, result: SearchResult) -> None:
    """Rank every observed cluster and retained box, best first, into ``result``."""
    result["ranked_clusters"] = sorted(
        state.observed_clusters, key=lambda cluster: cluster["score"], reverse=True
    )
    result["ranked_regions"] = sorted(
        state.observed_regions,
        key=lambda proposal: (proposal["score"], proposal["sam3_score"] or 0.0),
        reverse=True,
    )


def _attach_final_answer(state: RunState, result: SearchResult) -> None:
    """Answer the original question from the top-ranked region, when enabled."""
    if not state.settings.get("answer_question", True):
        return
    started = time.perf_counter()
    state.on_event(
        "answer_started",
        "Generate the final text answer from the top-ranked region.",
        {"question": state.question},
    )
    try:
        answer = answer_top_region(state, result)
    except Exception as exc:
        message = f"Final answer failed: {type(exc).__name__}: {exc}"
        result["warnings"] = [*result.get("warnings", []), message]
        state.on_event("answer_failed", message, {"error": message})
        return
    if answer is None:
        state.on_event(
            "answer_skipped",
            "No answerer or no ranked region; keeping the localization output only.",
            {"question": state.question},
        )
        return
    result["final_answer"] = answer
    state.on_event(
        "answer_completed",
        f"Final answer: {answer['answer']}",
        {
            "question": answer["question"],
            "answer": answer["answer"],
            "bbox": answer["bbox"],
            "crop_bbox": answer["crop_bbox"],
            "duration_seconds": time.perf_counter() - started,
            "model_id": answer["model_id"],
        },
    )


def _finalize_artifacts(state: RunState, result: SearchResult) -> None:
    """Index per-node galleries and write the final tree snapshot, if enabled."""
    if state.artifacts is None:
        return
    for folder in (state.artifacts / "nodes").glob("node_*"):
        stage_index(folder)
    save_tree(state.artifacts / "tree", result["nodes"], result["trace"])
    (state.artifacts / "index.html").write_text(TREE_INDEX_HTML)
