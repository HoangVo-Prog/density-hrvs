"""Compatibility facade for image conversion, SAM3 detection, and the CLI."""

from .core.imaging import crop_masked_image, to_pil_rgb
from .tools.legacy_cli import main
from .models.sam3 import exec_sam3_on_image, load_sam3_model

__all__ = [
    "crop_masked_image",
    "exec_sam3_on_image",
    "load_sam3_model",
    "main",
    "to_pil_rgb",
]


if __name__ == "__main__":
    raise SystemExit(main())
