"""Pure image-array helpers shared by detection, partitioning, and artifacts."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .contracts import BoundingBox, ChildRegion


def to_pil_rgb(image: NDArray) -> Image.Image:
    """Convert uint8 RGB or finite floating-point RGB in [0, 1] to PIL."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) < 1:
        raise ValueError("Expected a nonempty RGB image with shape (H, W, 3).")
    if array.dtype != np.uint8:
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Float RGB values must be in [0, 1].")
        array = np.rint(array * 255).astype(np.uint8)
    return Image.fromarray(array)


def crop_masked_image(
    image: NDArray,
    mask: NDArray,
) -> tuple[NDArray, BoundingBox]:
    """Crop to mask bounds, keeping black outside the mask; maxima are exclusive."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError("Image and 2D mask must match spatially.")
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("Cannot crop an empty mask.")
    y_min, y_max = int(rows.min()), int(rows.max()) + 1
    x_min, x_max = int(cols.min()), int(cols.max()) + 1
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    source = image[y_min:y_max, x_min:x_max]
    cropped_image = np.zeros_like(source)
    cropped_image[cropped_mask] = source[cropped_mask]
    return cropped_image, (x_min, y_min, x_max, y_max)


def region_pixels(image: NDArray, node: ChildRegion) -> NDArray[np.uint8]:
    """Render a node from original RGB pixels, retaining its exact black mask."""
    x1, y1, x2, y2 = node["bbox"]
    pixels = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    pixels[node["mask"]] = image[y1:y2, x1:x2][node["mask"]]
    return pixels


def clamp_global_box(
    node_bbox: BoundingBox,
    local_bbox: BoundingBox,
    image_size: tuple[int, int],
) -> BoundingBox | None:
    """Map a crop-local box into image coordinates and clip it to the image.

    Args:
        node_bbox: Parent region bounds in image coordinates.
        local_bbox: Box in the parent crop's coordinates.
        image_size: Full image ``(width, height)``.

    Returns:
        A nonempty image-coordinate box, or ``None`` when clipping removes it.
    """
    x0, y0, _, _ = node_bbox
    a, b, c, d = local_bbox
    width, height = image_size
    box: BoundingBox = (
        max(0, min(width, x0 + a)),
        max(0, min(height, y0 + b)),
        max(0, min(width, x0 + c)),
        max(0, min(height, y0 + d)),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box
