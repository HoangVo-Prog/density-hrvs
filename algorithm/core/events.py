"""Typed decision events and append-only console/file logging for a single run."""

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import TypedDict

from .contracts import (
    BoundingBox,
    RegionScoreComponents,
    SearchConfig,
    SearchUsage,
    SlicMetrics,
    VerifierScore,
)


class BranchCandidate(TypedDict):
    """One eligible child compared at a tree branch."""

    node_id: int
    visits: int
    quality: float  # Mean backed-up region reward before this decision.
    exploration_bonus: float
    estimated_remaining_seconds: float
    cost_penalty: float
    selection_index: float  # quality + exploration_bonus - cost_penalty.


class BackupUpdate(TypedDict):
    """Actual visit/Q update caused by a completed evaluation."""

    node_id: int
    visits_before: int
    visits_after: int
    quality_before: float
    quality_after: float


class SAM3CandidateRecord(TypedDict):
    """Native box metadata saved before thresholding or the proposal cap."""

    query_index: int
    normalized_bbox: tuple[float, float, float, float]  # Native normalized xyxy.
    confidence: float  # Sigmoid logit multiplied by presence, when supplied.
    retained: bool  # Included before NMS, mask resizing and invalid-box rejection.
    decision: str  # below_threshold, proposal_limit, or retained_before_nms.


class EventData(TypedDict, total=False):
    """Recognized event fields; image arrays and model weights never enter logs."""

    node_id: int
    parent_id: int
    selected_node: int
    iteration: int
    depth: int
    path: list[int]
    bbox: BoundingBox
    child_bboxes: list[BoundingBox]
    child_ids: list[int]
    excluded_node_ids: list[int]
    candidates: list[BranchCandidate]
    selection_reason: str
    backups: list[BackupUpdate]
    model_id: str
    duration_seconds: float
    image_size: tuple[int, int]
    feature_shape: list[int]
    selected_n_segments: int
    actual_superpixels: int
    slic_history: list[SlicMetrics]
    n_clusters: int
    used_spatial_fallback: bool
    split_reasons: dict[str, bool]
    qwen: VerifierScore
    score_components: RegionScoreComponents
    sam_candidates: list[SAM3CandidateRecord]
    boxes_before_nms: int
    boxes_after_nms: int
    threshold: float
    proposal_limit: int
    nms_iou: float
    candidate_rank: int
    sam3_score: float | None
    candidate_score: float
    view_type: str
    gate: str
    verified_box_count: int
    candidate_checks: int
    gated_regions: int
    sam3_first: bool
    config: SearchConfig
    query: str
    detector_prompt: str
    usage: SearchUsage
    reward: float
    stop_reason: str
    error: str
    recoverable: bool
    raw_responses: list[str]
    artifact_path: str


class RunEvent(TypedDict):
    """One chronological decision or operation, suitable for JSONL replay."""

    sequence: int
    timestamp_utc: str
    elapsed_seconds: float
    event: str
    message: str
    data: EventData


EventSink = Callable[[str, str, EventData], None]


def ignore_event(event: str, message: str, data: EventData) -> None:
    """No-op observer for callers that do not request decision logging."""


def create_event_logger(
    output: Path, console: Callable[[str], None] = print
) -> EventSink:
    """Append each event to events.jsonl and run.log, flushing after every write."""
    output.mkdir(parents=True, exist_ok=True)
    sequence = 0
    started = time.perf_counter()
    lock = threading.Lock()

    def emit(event: str, message: str, data: EventData) -> None:
        nonlocal sequence
        with lock:
            sequence += 1
            record: RunEvent = {
                "sequence": sequence,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "event": event,
                "message": message,
                "data": data,
            }
            line = f"[{sequence:04d} +{record['elapsed_seconds']:8.2f}s] {event}: {message}"
            with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            with (output / "run.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                if data:
                    stream.write("  " + json.dumps(data, ensure_ascii=False) + "\n")
            console(line)

    return emit
