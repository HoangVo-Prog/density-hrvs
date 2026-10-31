"""Region construction: DINO -> SLIC -> K-means -> geometry fallback."""

from collections.abc import Callable
import json
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
import torch

from ..core.contracts import (
    ChildRegion,
    SearchConfig,
    SearchNode,
    SlicClusters,
    SlicOptimization,
)
from ..core.diagnostics import (
    ClusterQuality,
    cluster_quality,
    save_image,
    save_labels,
    stage_index,
)
from ..core.events import EventSink, ignore_event
from ..core.imaging import region_pixels as region_pixels
from ..core.imaging import to_pil_rgb
from ..models.regions import (
    cluster_slic_regions,
    optimize_slic_n_segments,
)


FeatureExtractor = Callable[[NDArray], NDArray[np.float32]]


def _tight_child(mask: NDArray[np.bool_], parent: ChildRegion) -> ChildRegion:
    rows, cols = np.nonzero(mask)
    x1, y1, x2, y2 = (
        int(cols.min()),
        int(rows.min()),
        int(cols.max()) + 1,
        int(rows.max()) + 1,
    )
    px, py, _, _ = parent["bbox"]
    return {
        "bbox": (px + x1, py + y1, px + x2, py + y2),
        "mask": mask[y1:y2, x1:x2].copy(),
    }


def spatial_children(node: ChildRegion) -> list[ChildRegion]:
    """Partition active pixels into nonempty quadrants if semantic splitting stalls."""
    height, width = node["mask"].shape
    children: list[ChildRegion] = []
    for y1, y2 in ((0, height // 2), (height // 2, height)):
        for x1, x2 in ((0, width // 2), (width // 2, width)):
            mask = node["mask"][y1:y2, x1:x2].copy()
            if mask.any() and int(mask.sum()) < int(node["mask"].sum()):
                px, py, _, _ = node["bbox"]
                # Keep the full tile extent: tightening a fragmented parent mask
                # can create a 600x3 input even though its spatial tile is square.
                children.append(
                    {"bbox": (px + x1, py + y1, px + x2, py + y2), "mask": mask}
                )
    return children


def _should_skip_split(
    node: SearchNode, config: SearchConfig, on_event: EventSink
) -> bool:
    if (
        node["depth"] >= config["max_depth"]
        or int(node["mask"].sum()) < config["min_region_pixels"]
        or min(node["mask"].shape) < 32
    ):
        on_event(
            "split_skipped",
            f"N{node['node_id']}: depth, active-pixel or minimum-side stopping rule.",
            {"node_id": node["node_id"], "bbox": node["bbox"], "depth": node["depth"]},
        )
        return True
    return False


def _encode_crop(
    node: SearchNode,
    pixels: NDArray,
    extract_features: FeatureExtractor,
    on_event: EventSink,
) -> NDArray[np.float32]:
    on_event(
        "dino_started",
        f"Encode N{node['node_id']} at its current crop scale.",
        {
            "node_id": node["node_id"],
            "bbox": node["bbox"],
            "image_size": (pixels.shape[1], pixels.shape[0]),
        },
    )
    started = time.perf_counter()
    feature_map = extract_features(pixels)
    on_event(
        "dino_completed",
        f"DINO PCA features ready for N{node['node_id']}.",
        {
            "node_id": node["node_id"],
            "feature_shape": list(feature_map.shape),
            "duration_seconds": time.perf_counter() - started,
        },
    )
    return feature_map


def _fit_feature_map(
    feature_map: NDArray[np.float32], max_side: int
) -> NDArray[np.float32]:
    height, width = feature_map.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1:
        return feature_map
    return (
        torch.nn.functional.interpolate(
            torch.from_numpy(feature_map).permute(2, 0, 1).unsqueeze(0),
            size=(max(1, round(height * scale)), max(1, round(width * scale))),
            mode="nearest",
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )


def _save_feature_artifacts(
    artifact_dir: Path | None,
    node: SearchNode,
    pixels: NDArray,
    feature_map: NDArray[np.float32],
) -> None:
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_image(artifact_dir / "region_masked.jpg", pixels)
    np.savez_compressed(
        artifact_dir / "dino_pca.npz",
        features=feature_map,
        source_shape=np.asarray(pixels.shape),
        source_bbox=np.asarray(node["bbox"]),
    )
    save_image(
        artifact_dir / "dino_pca.png",
        np.rint(feature_map * 255).astype(np.uint8),
    )


def _segment_crop(
    node: SearchNode,
    feature_map: NDArray[np.float32],
    pixels: NDArray,
    config: SearchConfig,
    artifact_dir: Path | None,
    on_event: EventSink,
) -> SlicOptimization:
    def candidate(n_segments: int, labels: NDArray) -> None:
        on_event(
            "slic_candidate",
            f"N{node['node_id']}: evaluated requested SLIC count {n_segments}.",
            {
                "node_id": node["node_id"],
                "selected_n_segments": n_segments,
                "actual_superpixels": int(len(np.unique(labels))),
            },
        )
        if artifact_dir is not None:
            save_labels(
                artifact_dir / "slic_candidates",
                f"slic_{n_segments}",
                labels,
                pixels,
            )

    segmentation = optimize_slic_n_segments(
        feature_map,
        config["slic"],
        config["n_segments_values"],
        config["entropy_bins"],
        on_candidate=candidate,
    )
    selected_config = segmentation.get("config", config["slic"])
    on_event(
        "slic_selected",
        f"N{node['node_id']}: selected SLIC {selected_config['n_segments']} by minimum mean entropy.",
        {
            "node_id": node["node_id"],
            "selected_n_segments": selected_config["n_segments"],
            "actual_superpixels": len(segmentation["region_ids"]),
            "slic_history": segmentation.get("history", []),
        },
    )
    if artifact_dir is not None:
        save_labels(artifact_dir, "slic_selected", segmentation["labels"], pixels)
        (artifact_dir / "slic_metrics.json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in segmentation.items()
                    if key not in {"labels", "region_ids"}
                },
                default=lambda value: (
                    value.item() if isinstance(value, np.generic) else value.tolist()
                ),
                indent=2,
            )
        )
    return segmentation


def _save_cluster_artifacts(
    artifact_dir: Path | None,
    clusters: SlicClusters,
    pixels: NDArray,
) -> None:
    if artifact_dir is None:
        return
    save_labels(artifact_dir, "kmeans", clusters["labels"], pixels)
    np.savez_compressed(
        artifact_dir / "cluster_features.npz",
        region_features=clusters["region_features"],
        region_ids=clusters["region_ids"],
        region_clusters=clusters["region_clusters"],
    )
    display_rgb = np.asarray(
        Image.fromarray(pixels).resize(
            (clusters["labels"].shape[1], clusters["labels"].shape[0])
        )
    )
    for cluster_id in np.unique(clusters["labels"]):
        masked = np.where(
            (clusters["labels"] == cluster_id)[..., None],
            display_rgb,
            np.uint8(0),
        )
        save_image(artifact_dir / f"semantic_cluster_{cluster_id}.jpg", masked)


def _fallback_children(
    node: SearchNode,
    reason: str,
    message: str,
    on_event: EventSink,
) -> list[ChildRegion]:
    children = spatial_children(node)
    on_event(
        "split_completed",
        f"N{node['node_id']}: {message}",
        {
            "node_id": node["node_id"],
            "used_spatial_fallback": True,
            "split_reasons": {reason: True},
            "child_bboxes": [child["bbox"] for child in children],
        },
    )
    return children


def _cluster_crop(
    node: SearchNode,
    feature_map: NDArray[np.float32],
    segmentation: SlicOptimization,
    pixels: NDArray,
    config: SearchConfig,
    artifact_dir: Path | None,
    on_event: EventSink,
) -> tuple[SlicClusters | None, list[ChildRegion] | None]:
    count = min(config["branching_factor"], len(segmentation["region_ids"]))
    if count < 2:
        return None, _fallback_children(
            node, "too_few_groups", "too few SLIC groups; use spatial tiles.", on_event
        )
    try:
        clusters = cluster_slic_regions(
            feature_map,
            segmentation["labels"],
            n_clusters=count,
            spatial_weight=config.get("cluster_spatial_weight", 0.75),
        )
    except ValueError as exc:
        if "n_clusters must be between" not in str(exc):
            raise
        return None, _fallback_children(
            node,
            "too_few_distinct_features",
            "insufficient distinct features; use spatial tiles.",
            on_event,
        )
    _save_cluster_artifacts(artifact_dir, clusters, pixels)
    return clusters, None


def _children_from_clusters(
    node: SearchNode,
    clusters: SlicClusters,
    segmentation: SlicOptimization,
    original_width: int,
    original_height: int,
) -> list[ChildRegion]:
    cluster_labels = np.asarray(
        Image.fromarray(clusters["labels"].astype(np.int32)).resize(
            (original_width, original_height),
            Image.Resampling.NEAREST,
        )
    )
    slic_labels = np.asarray(
        Image.fromarray(segmentation["labels"].astype(np.int32)).resize(
            (original_width, original_height),
            Image.Resampling.NEAREST,
        )
    )
    children: list[ChildRegion] = []
    for cluster_id in np.unique(cluster_labels[node["mask"]]):
        mask = (cluster_labels == cluster_id) & node["mask"]
        if not mask.any() or int(mask.sum()) >= int(node["mask"].sum()):
            continue
        child = _tight_child(mask, node)
        child["region_ids"] = np.unique(slic_labels[mask]).tolist()
        children.append(child)
    return children


def _split_quality(
    children: list[ChildRegion],
    node: SearchNode,
    config: SearchConfig,
) -> tuple[bool, dict[str, bool], list[ClusterQuality]]:
    x1, y1, x2, y2 = node["bbox"]
    parent_area = (x2 - x1) * (y2 - y1)
    # Disconnected appearance groups can all span the full image. Such a split
    # removes pixels but gives no useful zoom; partition spatially in that case.
    no_zoom = all(
        (child["bbox"][2] - child["bbox"][0]) * (child["bbox"][3] - child["bbox"][1])
        >= 0.9 * parent_area
        for child in children
    )
    qualities = [cluster_quality(child) for child in children]
    poor_geometry = any(
        quality["occupancy"] < config.get("min_cluster_occupancy", 0.25)
        or quality["aspect"] > config.get("max_cluster_aspect", 6.0)
        or quality["largest_component_fraction"]
        < config.get("min_largest_component", 0.7)
        for quality in qualities
    )
    reasons = {
        "too_few_children": len(children) < 2,
        "no_zoom": no_zoom,
        "poor_geometry": poor_geometry,
    }
    return len(children) < 2 or no_zoom or poor_geometry, reasons, qualities


def _save_split_artifacts(
    artifact_dir: Path | None,
    image: NDArray,
    node: SearchNode,
    pixels: NDArray,
    selected_children: list[ChildRegion],
    qualities: list[ClusterQuality],
    fallback: bool,
    reasons: dict[str, bool],
) -> None:
    if artifact_dir is None:
        return
    (artifact_dir / "split.json").write_text(
        json.dumps(
            {
                "semantic_quality": qualities,
                "used_spatial_fallback": fallback,
                "reasons": reasons,
                "children": [cluster_quality(child) for child in selected_children],
            },
            indent=2,
        )
    )
    overlay = to_pil_rgb(pixels)
    draw = ImageDraw.Draw(overlay)
    px, py, _, _ = node["bbox"]
    for index, child in enumerate(selected_children):
        a, b, c, d = child["bbox"]
        draw.rectangle(
            (a - px, b - py, c - px - 1, d - py - 1),
            outline="cyan",
            width=max(2, overlay.width // 600),
        )
        draw.text((a - px, b - py), str(index), fill="red")
        save_image(
            artifact_dir / f"selected_child_{index}.jpg",
            region_pixels(image, child),
        )
    save_image(artifact_dir / "selected_children.jpg", np.asarray(overlay))
    stage_index(artifact_dir)


def _emit_split_completed(
    node: SearchNode,
    selected_children: list[ChildRegion],
    fallback: bool,
    reasons: dict[str, bool],
    n_clusters: int,
    on_event: EventSink,
) -> None:
    on_event(
        "split_completed",
        f"N{node['node_id']}: created {len(selected_children)} {'spatial' if fallback else 'semantic'} child regions.",
        {
            "node_id": node["node_id"],
            "n_clusters": n_clusters,
            "used_spatial_fallback": fallback,
            "split_reasons": reasons,
            "child_bboxes": [child["bbox"] for child in selected_children],
        },
    )


def semantic_children(
    image: NDArray,
    node: SearchNode,
    config: SearchConfig,
    extract_features: FeatureExtractor,
    artifact_dir: Path | None = None,
    on_event: EventSink = ignore_event,
) -> list[ChildRegion]:
    """Re-encode the current crop and partition only its active parent pixels.

    Args:
        image: Original full RGB image; all returned boxes use these coordinates.
        node: Current tight crop and local active mask.
        config: PCA resolution, SLIC candidates, and child/depth limits.
        extract_features: Fresh DINO PCA encoding for the rendered crop. Parent
            encodings are not reused as a substitute for higher-resolution views.
        artifact_dir: Optional per-node directory for lossless stage artifacts.
        on_event: Optional structured decision logger.

    Returns:
        Child regions in global coordinates, or an empty list at a stopping rule.
    """
    if _should_skip_split(node, config, on_event):
        return []
    pixels = region_pixels(image, node)
    feature_map = _encode_crop(node, pixels, extract_features, on_event)
    original_height, original_width = feature_map.shape[:2]
    feature_map = _fit_feature_map(feature_map, config["segmentation_max_side"])
    _save_feature_artifacts(artifact_dir, node, pixels, feature_map)
    segmentation = _segment_crop(
        node, feature_map, pixels, config, artifact_dir, on_event
    )
    clusters, fallback_children = _cluster_crop(
        node, feature_map, segmentation, pixels, config, artifact_dir, on_event
    )
    if fallback_children is not None:
        return fallback_children
    assert clusters is not None
    children = _children_from_clusters(
        node, clusters, segmentation, original_width, original_height
    )
    fallback, reasons, qualities = _split_quality(children, node, config)
    selected_children = spatial_children(node) if fallback else children
    _save_split_artifacts(
        artifact_dir,
        image,
        node,
        pixels,
        selected_children,
        qualities,
        fallback,
        reasons,
    )
    _emit_split_completed(
        node,
        selected_children,
        fallback,
        reasons,
        min(config["branching_factor"], len(segmentation["region_ids"])),
        on_event,
    )
    return selected_children
