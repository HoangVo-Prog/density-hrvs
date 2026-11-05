"""Seed the tree from tiled SAM3 candidates reranked by the local verifier."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.contracts import ChildRegion, SAM3Detection
from ..core.errors import LocalBudgetExceeded
from ..models.evidence import build_proposal_evidence
from .evaluation import score_crop
from .region_scoring import distinct_detections
from .runtime import RunState


def _tiles(width: int, height: int, count: int, overlap: float):
    for gy in range(count):
        for gx in range(count):
            x0 = int(max(0, (gx / count - overlap) * width))
            x1 = int(min(width, ((gx + 1) / count + overlap) * width))
            y0 = int(max(0, (gy / count - overlap) * height))
            y1 = int(min(height, ((gy + 1) / count + overlap) * height))
            if x1 > x0 and y1 > y0:
                yield x0, y0, x1, y1


def discover_seed_regions(state: RunState) -> list[ChildRegion]:
    """Run a tiled SAM3 scan and return the top Qwen-ranked seed regions."""
    settings = state.settings
    if settings.get("discovery_mode", "off") != "sam3_qwen":
        return []
    if not settings.get("sam3_first", True):
        return []
    tiles = settings.get("discovery_tiles", 3)
    candidate_limit = settings.get("discovery_candidates", 5)
    seed_limit = settings.get("discovery_seeds", 3)
    overlap = settings.get("discovery_overlap", 0.10)
    if candidate_limit <= 0 or seed_limit <= 0:
        return []

    height, width = state.rgb.shape[:2]
    prompt = settings["sam_prompt"] or state.query
    state.on_event(
        "discovery_started",
        f"Discovery: scan {tiles}x{tiles} SAM3 tiles for {prompt!r}.",
        {"tiles": tiles, "overlap": overlap, "prompt": prompt},
    )
    detections: list[SAM3Detection] = []
    for x0, y0, x1, y1 in _tiles(width, height, tiles, overlap):
        found = state.detector(state.rgb[y0:y1, x0:x1], prompt)
        for detection in found:
            a, b, c, d = detection["bbox"]
            bbox = (x0 + a, y0 + b, min(width, x0 + c), min(height, y0 + d))
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                detections.append({**detection, "bbox": bbox})

    distinct = distinct_detections(detections, settings)
    ranked = []
    for detection in distinct[:candidate_limit]:
        try:
            view = build_proposal_evidence(
                state.source_image,
                detection["bbox"],
                max_side=settings.get("verification_max_side", 1536),
                context_padding=settings["context_padding"],
                detail_padding=settings["detail_padding"],
            )
        except ValueError:
            continue
        try:
            rating = score_crop(
                state,
                Image.fromarray(view["image"]),
                state.query,
                "context_detail",
            )
        except LocalBudgetExceeded:
            break
        ranked.append({"rating": rating, "detection": detection})
    ranked.sort(
        key=lambda item: (
            item["rating"]["score"],
            item["detection"]["sam3_score"] or 0.0,
        ),
        reverse=True,
    )

    seeds: list[ChildRegion] = []
    serializable = []
    for item in ranked:
        detection = item["detection"]
        x0, y0, x1, y1 = detection["bbox"]
        serializable.append(
            {
                "bbox": detection["bbox"],
                "sam3_score": detection["sam3_score"],
                "qwen": item["rating"],
                "seed": len(seeds) < seed_limit,
            }
        )
        if len(seeds) >= seed_limit:
            continue
        seeds.append(
            {
                "bbox": (x0, y0, x1, y1),
                "mask": np.ones((y1 - y0, x1 - x0), dtype=bool),
            }
        )
        state.on_event(
            "discovery_candidate",
            f"Discovery seed {len(seeds)}: {detection['bbox']} Qwen {item['rating']['score']:.3f}, {item['rating']['answer']}.",
            {
                "seed_rank": len(seeds),
                "bbox": detection["bbox"],
                "sam3_score": detection["sam3_score"],
                "qwen": item["rating"],
            },
        )

    state.usage["candidate_checks"] = state.usage.get("candidate_checks", 0) + len(
        ranked
    )
    if state.artifacts is not None:
        folder = Path(state.artifacts)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "discovery.json").write_text(json.dumps(serializable, indent=2))
    state.on_event(
        "discovery_completed",
        f"Discovery ranked {len(ranked)} candidates and seeded {len(seeds)} regions.",
        {
            "candidate_count": len(ranked),
            "seed_count": len(seeds),
            "seed_bboxes": [seed["bbox"] for seed in seeds],
            "usage": state.usage.copy(),
        },
    )
    return seeds
