"""Typed request, per-run state, budgets, and shared caches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..core.contracts import (
    BoundingBox,
    CandidateCheck,
    ChildRegion,
    RankedCluster,
    RankedRegion,
    SAM3Detection,
    SearchConfig,
    SearchUsage,
    VerifierScore,
)
from ..core.errors import APIBudgetExceeded, LocalBudgetExceeded
from ..core.events import EventSink, ignore_event
from ..models.scoring import ScoringView

if TYPE_CHECKING:
    from ..models.qwen_scoring import LocalScorer

FeatureExtractor = Callable[[NDArray], NDArray[np.float32]]
Detector = Callable[[NDArray, str], list[SAM3Detection]]
Scorer = Callable[[Image.Image, str], VerifierScore]
CostStage = Literal["score", "detect", "split"]


@dataclass(frozen=True)
class SearchRequest:
    """Typed inputs for one visual search; callables are injected for tests."""

    image: NDArray
    query: str
    config: SearchConfig | None = None
    scorer: Scorer | None = None
    detector: Detector | None = None
    feature_extractor: FeatureExtractor | None = None
    progress: Callable[[str], None] = print
    local_scorer: LocalScorer | None = None
    artifact_dir: str | Path | None = None
    on_event: EventSink = ignore_event


@dataclass
class NodeContext:
    """Mutable artifact target for the region currently being evaluated."""

    node_id: int = -1
    folder: Path | None = None


@dataclass
class RunState:
    """All mutable state for one search, replacing hidden closure variables."""

    query: str
    settings: SearchConfig
    usage: SearchUsage
    source_image: Image.Image
    rgb: NDArray[np.uint8]
    artifacts: Path | None
    on_event: EventSink
    progress: Callable[[str], None]
    custom_scorer: Scorer | None
    local_scorer: LocalScorer | None
    feature_extractor: FeatureExtractor
    detector: Detector
    context: NodeContext = field(default_factory=NodeContext)
    seed_children: list[ChildRegion] | None = None
    started_at: float = field(default_factory=time.perf_counter)
    score_cache: dict[str, VerifierScore] = field(default_factory=dict)
    detection_cache: dict[int, list[SAM3Detection]] = field(default_factory=dict)
    checked_cache: dict[int, list[CandidateCheck]] = field(default_factory=dict)
    split_cache: dict[int, list[ChildRegion]] = field(default_factory=dict)
    observed_regions: list[RankedRegion] = field(default_factory=list)
    observed_clusters: list[RankedCluster] = field(default_factory=list)
    seen_proposals: dict[tuple[BoundingBox, str], int] = field(default_factory=dict)
    score_samples: list[float] = field(default_factory=list)
    detect_samples: list[float] = field(default_factory=list)
    split_samples: list[float] = field(default_factory=list)

    def charge_request(self) -> None:
        """Bill one physical request or raise before sending it."""
        if (
            self.settings["max_api_calls"]
            and self.usage["api_calls"] >= self.settings["max_api_calls"]
        ):
            raise APIBudgetExceeded(
                "HTTP request budget exhausted before sending the next request."
            )
        self.usage["api_calls"] += 1

    def charge_local(self) -> None:
        """Bill one local generation or raise before running it."""
        count = self.usage.get("local_score_calls", 0)
        budget = self.settings.get("max_local_scores", 32)
        if budget and count >= budget:
            raise LocalBudgetExceeded("Local generation budget exhausted")
        self.usage["local_score_calls"] = count + 1

    def node_folder(self, node_id: int) -> Path | None:
        """Return the optional per-node artifact directory."""
        if self.artifacts is None:
            return None
        return self.artifacts / "nodes" / f"node_{node_id:04d}"

    def view_key(
        self,
        crop: Image.Image,
        view_type: ScoringView = "region",
        description: str | None = None,
    ) -> str:
        """Hash the exact verifier input and query text for reuse."""
        digest = hashlib.sha256(
            (self.query if description is None else description).encode()
        )
        digest.update(view_type.encode("ascii"))
        digest.update(str(crop.size).encode("ascii"))
        digest.update(crop.tobytes())
        return digest.hexdigest()

    def record_cost(self, stage: CostStage, seconds: float) -> None:
        """Record one observed operation duration."""
        values = getattr(self, f"{stage}_samples")
        values.append(seconds)

    def mean_cost(self, stage: CostStage, default: float) -> float:
        """Return a stable observed mean, falling back before the first sample."""
        values = getattr(self, f"{stage}_samples")
        return max(0.001, sum(values) / len(values)) if values else default
