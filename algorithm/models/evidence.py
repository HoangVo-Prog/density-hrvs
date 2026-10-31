"""Deterministic context/detail views for visual verification of small objects."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..core.contracts import (
    BoundingBox,
    CandidatePrompt,
    EvidenceInfo,
    ProposalEvidence,
)


def evidence_info(view: ProposalEvidence) -> EvidenceInfo:
    """Return the serializable coordinate mapping without the RGB pixel array."""
    return {
        "source_bbox": view["source_bbox"],
        "context_bbox": view["context_bbox"],
        "detail_bbox": view["detail_bbox"],
        "canvas_size": view["canvas_size"],
        "source_detail_size": view["source_detail_size"],
        "rendered_detail_size": view["rendered_detail_size"],
        "context_target_bbox": view["context_target_bbox"],
        "detail_target_bbox": view["detail_target_bbox"],
    }


def build_region_candidate_sheet(
    image: Image.Image,
    candidates: list[CandidatePrompt],
    *,
    max_side: int = 1536,
) -> Image.Image:
    """Draw numbered SAM3 boxes on the whole cluster image, preserving context."""
    if max_side < 512:
        raise ValueError("Region sheet max side must be at least 512 pixels.")
    source = image.convert("RGB")
    scale = min(1.0, max_side / max(source.width, source.height))
    if scale < 1.0:
        sheet = source.resize(
            (
                max(1, round(source.width * scale)),
                max(1, round(source.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        sheet = source.copy()
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    label_size = max(12, round(max_side / 80))
    face = (
        ImageFont.truetype(str(font_path), label_size)
        if font_path.exists()
        else ImageFont.load_default(size=label_size)
    )
    colors = ("#00d9ff", "#ff453a", "#ffae00", "#cb75ff", "#00ff85", "#ff5ca8")
    stroke = max(2, round(max_side / 512))
    for candidate, color in zip(candidates, colors):
        x1, y1, x2, y2 = candidate["bbox"]
        box = (
            round(x1 * scale),
            round(y1 * scale),
            round(x2 * scale),
            round(y2 * scale),
        )
        draw.rectangle(
            (box[0], box[1], max(box[0], box[2] - 1), max(box[1], box[3] - 1)),
            outline=color,
            width=stroke,
        )
        label = f"{candidate['index']}: {candidate['label']}"
        if candidate["confidence"] is not None:
            label += f" {candidate['confidence']:.2f}"
        if len(label) > 36:
            label = label[:33].rstrip() + "..."
        text_box = draw.textbbox((0, 0), label, font=face)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = min(max(0, box[0]), max(0, sheet.width - text_width - 6))
        text_y = max(0, box[1] - text_height - 6)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 4, text_y + text_height + 4),
            fill="#111827",
        )
        draw.text((text_x + 2, text_y + 2), label, font=face, fill=color)
    return sheet


def build_proposal_evidence(
    original: Image.Image,
    bbox: BoundingBox,
    *,
    max_side: int = 1536,
    context_padding: int = 384,
    detail_padding: int = 16,
) -> ProposalEvidence:
    """Compose one model input with marked context and an enlarged target detail.

    Args:
        original: Unmasked original RGB image; no target annotations are needed.
        bbox: SAM3 candidate bounds in original pixels, with exclusive maxima.
        max_side: Uploaded sheet width (minimum 512); height is two-thirds of it.
        context_padding: Minimum margin on each side; also at least three times
            the larger target dimension to retain relational context.
        detail_padding: Minimum detail margin; also at least 25% of target size.

    Returns:
        A lossless RGB sheet, exact source windows, rendered sizes and target-box
        mappings. Both views preserve aspect ratio. Enlargement interpolates
        existing pixels and does not invent object detail.
    """
    if max_side < 512 or context_padding < 0 or detail_padding < 0:
        raise ValueError(
            "Evidence width must be >=512 and padding must be nonnegative."
        )
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= original.width and 0 <= y1 < y2 <= original.height):
        raise ValueError("Candidate box must lie within the original image.")
    source = original.convert("RGB")
    width, height = x2 - x1, y2 - y1

    def padded(margin: int) -> BoundingBox:
        return (
            max(0, x1 - margin),
            max(0, y1 - margin),
            min(source.width, x2 + margin),
            min(source.height, y2 + margin),
        )

    context_box = padded(max(context_padding, 3 * max(width, height)))
    detail_box = padded(max(detail_padding, round(max(width, height) * 0.25)))
    canvas_width, canvas_height = max_side, round(max_side * 2 / 3)
    gap = max(8, round(max_side / 96))
    header = max(32, round(max_side / 22))
    detail_width = round(max_side * 0.36)
    context_width = canvas_width - detail_width - 3 * gap
    panel_height = canvas_height - header - 3 * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#111827")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    label_size = max(14, round(max_side / 55))
    face = (
        ImageFont.truetype(str(font_path), label_size)
        if font_path.exists()
        else ImageFont.load_default(size=label_size)
    )
    target_color = "#00d9ff"

    def panel(
        box: BoundingBox, left: int, panel_width: int, title: str
    ) -> tuple[BoundingBox, tuple[int, int]]:
        crop = source.crop(box)
        # contain() explicitly enlarges tiny crops, unlike thumbnail().
        rendered = ImageOps.contain(
            crop, (panel_width, panel_height), Image.Resampling.LANCZOS
        )
        px = left + (panel_width - rendered.width) // 2
        py = header + gap + (panel_height - rendered.height) // 2
        draw.rectangle(
            (left, header + gap, left + panel_width - 1, canvas_height - gap),
            fill="#1f2937",
        )
        canvas.paste(rendered, (px, py))
        sx, sy = rendered.width / crop.width, rendered.height / crop.height
        mapped: BoundingBox = (
            px + round((x1 - box[0]) * sx),
            py + round((y1 - box[1]) * sy),
            px + round((x2 - box[0]) * sx),
            py + round((y2 - box[1]) * sy),
        )
        stroke = max(2, round(max_side / 512))
        draw.rectangle(
            (mapped[0], mapped[1], mapped[2] - 1, mapped[3] - 1),
            outline=target_color,
            width=stroke,
        )
        draw.text((left, gap), title, font=face, fill="white")
        return mapped, rendered.size

    context_target, _ = panel(context_box, gap, context_width, "CONTEXT: boxed target")
    detail_target, rendered_size = panel(
        detail_box, context_width + 2 * gap, detail_width, "TARGET DETAIL: enlarged"
    )
    return {
        "image": np.asarray(canvas),
        "source_bbox": bbox,
        "context_bbox": context_box,
        "detail_bbox": detail_box,
        "canvas_size": canvas.size,
        "source_detail_size": (
            detail_box[2] - detail_box[0],
            detail_box[3] - detail_box[1],
        ),
        "rendered_detail_size": rendered_size,
        "context_target_bbox": context_target,
        "detail_target_bbox": detail_target,
    }
