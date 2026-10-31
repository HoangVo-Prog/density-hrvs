"""DINOv3 PCA and semantic SLIC partitions, shared by the notebook and runner.

The feature input intentionally matches the notebook: displayed three-channel
PCA colors, rather than a text-aligned embedding or a similarity score.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.segmentation import slic
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor

from ..core.contracts import SlicClusters, SlicConfig, SlicMetrics, SlicOptimization


def exec_slic_on_image(
    image: NDArray,
    slic_config: SlicConfig,
) -> NDArray[np.intp]:
    """Return a region ID for every pixel in an RGB image."""
    return slic(
        image,
        **slic_config,
        start_label=1,
        channel_axis=-1,
    )


DINOV3_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"


def load_dinov3(
    device: str = "auto",
    local_files_only: bool = True,
) -> tuple[PreTrainedModel, BaseImageProcessor]:
    """Load the project's DINOv3 checkpoint and image processor."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(
        DINOV3_ID,
        revision=DINOV3_REVISION,
        local_files_only=local_files_only,
    )
    model = (
        AutoModel.from_pretrained(
            DINOV3_ID,
            revision=DINOV3_REVISION,
            dtype=torch.float32,
            local_files_only=local_files_only,
        )
        .to(device)
        .eval()
    )

    return model, processor


@torch.inference_mode()
def exec_dinov3_pca_on_image(
    image: NDArray,
    model: PreTrainedModel,
    processor: BaseImageProcessor,
    max_side: int = 1024,
) -> NDArray[np.float32]:
    """Return an H×W×3 PCA visualization in [0, 1].

    Input: RGB uint8 image in [0, 255], or floating-point RGB in [0, 1].
    Output: original image dimensions, with nearest-neighbor patch colors.
    """
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating-point images must be finite and in [0, 1].")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("Expected uint8 or floating-point RGB input.")

    original_h, original_w = image.shape[:2]
    patch_size = model.config.patch_size

    if max_side < 2 * patch_size:
        raise ValueError("max_side must be at least twice the patch size.")

    scale = min(1.0, max_side / max(original_h, original_w))
    height = max(patch_size, round(original_h * scale / patch_size) * patch_size)
    width = max(patch_size, round(original_w * scale / patch_size) * patch_size)

    parameter = next(model.parameters())
    inputs = processor(
        images=image,
        size={"height": height, "width": width},
        do_center_crop=False,
        return_tensors="pt",
    )  # type:ignore
    inputs = {
        key: value.to(
            device=parameter.device,
            dtype=parameter.dtype if value.is_floating_point() else value.dtype,
        )
        for key, value in inputs.items()
    }

    model.eval()
    output = model(**inputs, return_dict=True)

    offset = 1 + model.config.num_register_tokens
    features = output.last_hidden_state[0, offset:].float().cpu().numpy()

    processed_h, processed_w = inputs["pixel_values"].shape[-2:]
    grid_h = processed_h // patch_size
    grid_w = processed_w // patch_size

    if features.shape[0] != grid_h * grid_w:
        raise ValueError("Patch token count does not match the image grid.")
    if min(features.shape) < 3:
        raise ValueError(
            "At least three patch tokens and feature channels are required."
        )

    components = PCA(
        n_components=3,
        svd_solver="randomized",
        random_state=0,
    ).fit_transform(features)

    low, high = np.percentile(components, [1, 99], axis=0)
    colors = np.clip(
        (components - low) / np.maximum(high - low, 1e-9),
        0,
        1,
    ).astype(np.float32)

    native_map = colors.reshape(grid_h, grid_w, 3)
    tensor = torch.from_numpy(native_map).permute(2, 0, 1).unsqueeze(0)

    return (
        F.interpolate(
            tensor,
            size=(original_h, original_w),
            mode="nearest",
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )


def optimize_slic_n_segments(
    image: NDArray,
    slic_config: SlicConfig,
    n_segments_values: Sequence[int] = (100, 200, 400, 600, 800, 1000),
    entropy_bins: int = 8,
    on_candidate: Callable[[int, NDArray], None] | None = None,
) -> SlicOptimization:
    """Select n_segments by minimizing mean per-region color entropy.

    Accepts RGB uint8 [0, 255] or float RGB [0, 1].
    Does not modify the supplied configuration.
    """
    rgb = np.asarray(image)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32)

    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Image values must be finite and in [0, 1].")
    if entropy_bins < 2:
        raise ValueError("entropy_bins must be at least 2.")

    candidates = sorted(set(n_segments_values))
    if not candidates or any(n < 1 for n in candidates):
        raise ValueError("Provide positive n_segments values.")

    # Use identical color bins for every candidate segmentation.
    quantized = np.minimum(
        (rgb * entropy_bins).astype(np.int32),
        entropy_bins - 1,
    )
    color_codes = (
        quantized[..., 0] * entropy_bins**2
        + quantized[..., 1] * entropy_bins
        + quantized[..., 2]
    ).ravel()

    best_result: SlicOptimization | None = None
    history: list[SlicMetrics] = []

    for n_segments in candidates:
        config: SlicConfig = {
            **slic_config,
            "n_segments": n_segments,
        }
        labels = exec_slic_on_image(rgb, config)
        if on_candidate is not None:
            on_candidate(n_segments, labels)

        region_ids, region_indices, region_sizes = np.unique(
            labels,
            return_inverse=True,
            return_counts=True,
        )
        region_indices = region_indices.ravel()

        # Count observed (region, color-bin) combinations without a dense table.
        pair_codes = region_indices * entropy_bins**3 + color_codes
        pairs, counts = np.unique(pair_codes, return_counts=True)
        pair_regions = pairs // entropy_bins**3

        probabilities = counts / region_sizes[pair_regions]
        region_entropies = np.bincount(
            pair_regions,
            weights=-probabilities * np.log2(probabilities),
            minlength=len(region_ids),
        ).astype(np.float64)

        metrics: SlicMetrics = {
            "n_segments": n_segments,
            "actual_regions": len(region_ids),
            "mean_entropy": float(region_entropies.mean()),
            "p90_entropy": float(np.percentile(region_entropies, 90)),
            "max_entropy": float(region_entropies.max()),
        }
        history.append(metrics)

        # Candidates are ascending: exact ties favor fewer requested segments.
        if (
            best_result is None
            or metrics["mean_entropy"] < best_result["metrics"]["mean_entropy"]
        ):
            best_result = {
                "labels": labels,
                "config": config,
                "metrics": metrics,
                "region_ids": region_ids,
                "region_entropies": region_entropies,
                "history": history,
            }

    if best_result is None:
        raise RuntimeError("No SLIC candidate was evaluated.")
    best_result["history"] = history
    return best_result


def cluster_slic_regions(
    image: NDArray,
    slic_labels: NDArray,
    n_clusters: int = 6,
    random_state: int = 42,
    spatial_weight: float = 0.0,
) -> SlicClusters:
    """Group SLIC regions using their mean RGB/PCA visualization colors."""
    rgb = np.asarray(image)
    rgb = (
        rgb.astype(np.float32) / 255.0
        if rgb.dtype == np.uint8
        else rgb.astype(np.float32)
    )

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    if rgb.shape[:2] != slic_labels.shape:
        raise ValueError("Image and SLIC labels must have matching dimensions.")
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Image values must be finite and in [0, 1].")

    region_ids, inverse, counts = np.unique(
        slic_labels,
        return_inverse=True,
        return_counts=True,
    )
    inverse = inverse.ravel()
    pixels = rgb.reshape(-1, 3)

    region_features = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=pixels[:, channel],
                minlength=len(region_ids),
            )
            / counts
            for channel in range(3)
        ]
    )
    if spatial_weight < 0 or not np.isfinite(spatial_weight):
        raise ValueError("spatial_weight must be nonnegative and finite")
    if spatial_weight:
        rows, columns = np.indices(slic_labels.shape)
        centroids = np.column_stack(
            [
                np.bincount(inverse, weights=columns.ravel(), minlength=len(region_ids))
                / counts
                / max(1, slic_labels.shape[1] - 1),
                np.bincount(inverse, weights=rows.ravel(), minlength=len(region_ids))
                / counts
                / max(1, slic_labels.shape[0] - 1),
            ]
        )
        region_features = np.column_stack((region_features, spatial_weight * centroids))

    distinct_features = np.unique(region_features, axis=0).shape[0]
    if not 2 <= n_clusters <= distinct_features:
        raise ValueError(f"n_clusters must be between 2 and {distinct_features}.")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
    )
    model.set_params(n_init=10)
    region_clusters = model.fit_predict(region_features)

    # Assign every pixel its superpixel's cluster.
    cluster_labels = region_clusters[inverse].reshape(slic_labels.shape)

    return {
        "labels": cluster_labels,
        "region_ids": region_ids,
        "region_clusters": region_clusters,
        "region_features": region_features,
        "model": model,
    }
