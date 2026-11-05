"""Shared vision model loaders for the standalone density-hrvs checkout."""

from __future__ import annotations

import os

import torch
from transformers import Sam3Model, Sam3Processor


SAM3_MODEL = os.environ.get("SAM3_MODEL", "facebook/sam3")
SAM3_REVISION = os.environ.get("SAM3_REVISION") or None


def load_sam3(device: str = "auto", dtype: torch.dtype = torch.float32):
    """Load the pinned SAM3 image detector and its processor."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained(
        SAM3_MODEL,
        revision=SAM3_REVISION,
        local_files_only=True,
    )
    model = (
        Sam3Model.from_pretrained(
            SAM3_MODEL,
            revision=SAM3_REVISION,
            dtype=dtype,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    return model, processor
