"""Exact bounding-box overlays and post-run inspection; no model/API calls."""

import argparse
import html
import json
import math
import os
from pathlib import Path
import textwrap
from typing import NotRequired, TypedDict, cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..core.contracts import BoundingBox, ProposalReport, SearchReport
from ..core.data import load_dataset_image
from ..core.geometry import box_iou


class CandidateInspection(TypedDict):
    """One prediction compared with the supplied annotation after search finishes."""

    candidate_id: int
    bbox: BoundingBox
    sam3_score: float | None
    verifier_score: float  # Qwen rating for this candidate, or the legacy region.
    verifier_answer: str
    verifier_reason: str
    candidate_score: float  # Fused proposal score, or the legacy region reward.
    target_iou: float
    full_image: str
    inspection_image: str
    model_input_image: NotRequired[str]  # Exact uploaded PNG, when saved by the run.
    region_score: NotRequired[
        float
    ]  # Combined region reward, not an individual-box rating.
    scoring_scope: NotRequired[str]
    candidate_rank: NotRequired[int]
    candidate_verified: NotRequired[bool]


class InspectionSummary(TypedDict):
    """Serializable inspection summary; ground truth is used only for evaluation."""

    image_id: str
    question_id: str
    question: str
    annotated_answer: str
    target_bbox: tuple[float, float, float, float]
    stop_reason: str
    api_calls: int
    elapsed_seconds: float
    reported_cost_usd: float
    best_candidate_id: int
    best_target_iou: float
    candidates: list[CandidateInspection]


COLORS = ("#ff453a", "#00bfff", "#ffae00", "#cb75ff", "#ff5ca8", "#00d4be")


def model_status(proposal: ProposalReport) -> str:
    """Describe candidate and verifier roles without implying calibrated confidence."""
    detector = f"SAM3 {proposal['sam3_score'] or 0:.3f}"
    model_id = proposal["verifier"].get("model_id", "")
    verifier = "Qwen3-VL-2B" if "Qwen" in model_id else model_id or "not-run"
    if proposal.get("scoring_scope") == "region":
        return (
            f"{detector} | Region reward {proposal['score']:.3f} | "
            f"{verifier} region {100 * proposal['verifier']['score']:.0f}/100"
        )
    rank = proposal.get("candidate_rank")
    prefix = f"Candidate {rank} | " if rank is not None else ""
    return (
        f"{detector} | {prefix}fused {proposal['score']:.3f} | "
        f"{verifier} {100 * proposal['verifier']['score']:.0f}/100 "
        f"{proposal['verifier']['answer']}"
    )


def font(size: int):
    """Use an installed readable font, with Pillow's bundled fallback."""
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default(size=size)
    )


def mark_box(
    image: Image.Image,
    box: tuple[float, float, float, float],
    label: str,
    color: str,
    *,
    label_offset: int = 0,
) -> None:
    """Draw the exact box plus an offset label without moving its coordinates."""
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = (
        math.floor(box[0]),
        math.floor(box[1]),
        math.ceil(box[2]),
        math.ceil(box[3]),
    )
    draw.rectangle((x1 - 2, y1 - 2, x2 + 1, y2 + 1), outline="black", width=3)
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=3)
    face = font(28)
    bounds = draw.textbbox((0, 0), label, font=face)
    tw, th = bounds[2] - bounds[0] + 16, bounds[3] - bounds[1] + 18
    lx = max(0, min(x1, image.width - tw))
    ly = max(0, min(y2 + 8 + label_offset, image.height - th))
    draw.rectangle((lx, ly, lx + tw, ly + th), fill="black")
    draw.text((lx + 8, ly + 4), label, font=face, fill=color)


def _legacy_card(
    original: Image.Image,
    run: Path,
    proposal: ProposalReport,
    index: int,
    color: str,
    overlap: float,
    image_id: str,
) -> Image.Image:
    """Render historical runs that scored only a tight crop."""
    bbox = proposal["bbox"]
    padding = max(256, 3 * (bbox[2] - bbox[0]), 3 * (bbox[3] - bbox[1]))
    context_box = (
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(original.width, bbox[2] + padding),
        min(original.height, bbox[3] + padding),
    )
    context = original.crop(context_box)
    cx, cy = context_box[:2]
    local = (
        float(bbox[0] - cx),
        float(bbox[1] - cy),
        float(bbox[2] - cx),
        float(bbox[3] - cy),
    )
    mark_box(context, local, f"P{index}", color)
    card = Image.new("RGB", (1500, 950), "#111827")
    draw = ImageDraw.Draw(card)
    draw.text(
        (28, 18),
        f"{image_id} / Candidate P{index} | Target IoU {overlap:.3f}",
        font=font(32),
        fill="white",
    )
    draw.text(
        (28, 76),
        "Original-image context (inspection only)",
        font=font(22),
        fill="#cbd5e1",
    )
    draw.text(
        (1000, 76), "Actual scored crop (enlarged)", font=font(22), fill="#cbd5e1"
    )
    card.paste(ImageOps.pad(context, (940, 650), color="#1f2937"), (28, 115))
    with Image.open(run / proposal["image_path"]) as crop:
        card.paste(
            ImageOps.pad(crop.convert("RGB"), (460, 650), color="#1f2937"), (1000, 115)
        )
    status = model_status(proposal)
    draw.text((28, 790), status, font=font(27), fill=color)
    for line_number, line in enumerate(
        textwrap.wrap(proposal["verifier"]["reason"], 110)[:3]
    ):
        draw.text((28, 840 + 29 * line_number), line, font=font(23), fill="white")
    return card


def _uploaded_card(
    image_path: Path,
    proposal: ProposalReport,
    index: int,
    color: str,
    overlap: float,
    image_id: str,
) -> Image.Image:
    """Show the exact uploaded evidence pixels, with a separate report header."""
    with Image.open(image_path) as source:
        uploaded = source.convert("RGB")
    width = max(1500, uploaded.width)
    card = Image.new("RGB", (width, uploaded.height + 255), "#111827")
    card.paste(uploaded, ((width - uploaded.width) // 2, 80))
    draw = ImageDraw.Draw(card)
    draw.text(
        (24, 15),
        f"{image_id} / P{index} | IoU {overlap:.3f} | Exact verifier input below",
        font=font(30),
        fill="white",
    )
    status = model_status(proposal)
    draw.text((24, uploaded.height + 100), status, font=font(27), fill=color)
    for i, line in enumerate(textwrap.wrap(proposal["verifier"]["reason"], 110)[:3]):
        draw.text(
            (24, uploaded.height + 150 + i * 28), line, font=font(22), fill="white"
        )
    return card


def _region_card(
    image_path: Path,
    proposal: ProposalReport,
    index: int,
    overlap: float,
    image_id: str,
) -> Image.Image:
    """Show the whole region assessed by Qwen, without implying box verification."""
    with Image.open(image_path) as source:
        region = ImageOps.contain(source.convert("RGB"), (1536, 1024))
    card = Image.new("RGB", (1536, region.height + 250), "#111827")
    card.paste(region, ((1536 - region.width) // 2, 90))
    draw = ImageDraw.Draw(card)
    draw.text(
        (24, 15),
        f"{image_id} / P{index} IoU {overlap:.3f} | Whole-region Qwen input",
        font=font(27),
        fill="white",
    )
    draw.text(
        (24, 52),
        "Display resized. This box was not individually verified by Qwen.",
        font=font(22),
        fill="#67e8f9",
    )
    draw.text(
        (24, region.height + 105), model_status(proposal), font=font(22), fill="white"
    )
    for n, line in enumerate(textwrap.wrap(proposal["verifier"]["reason"], 115)[:3]):
        draw.text((24, region.height + 140 + n * 27), line, font=font(21), fill="white")
    return card


def make_inspection(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    data_dir: str | Path,
    image_id: str,
    question_id: str,
) -> InspectionSummary:
    """Map saved predictions to the source image and export comparison pictures.

    Args:
        run_dir: Completed algorithm output containing results.json and crop files.
        output_dir: New folder for full-resolution overlays, cards and index.html.
        data_dir: Image ZIP and HRVS annotation directory, used only after the run.
        image_id: Original dataset image ID, such as 0063.
        question_id: Exact annotated question used for post-run IoU comparison.
    """
    run = Path(run_dir)
    output = Path(output_dir)
    (output / "full_images").mkdir(parents=True, exist_ok=True)
    (output / "candidate_views").mkdir(exist_ok=True)
    report = cast(SearchReport, json.loads((run / "results.json").read_text()))
    matches = []
    for path in sorted(Path(data_dir).glob("hrvs_bbox_samples_*.json")):
        samples = json.loads(path.read_text())
        matches.extend(
            row
            for row in samples
            if row.get("question_id") == question_id
            and str(row.get("image_id")) == image_id
        )
    if not matches:
        raise ValueError(f"No target annotation found for {question_id}.")
    sample = matches[0]
    target = sample["target_bbox"]
    truth = (
        float(target["x"]),
        float(target["y"]),
        float(target["x"] + target["width"]),
        float(target["y"] + target["height"]),
    )
    original = load_dataset_image(data_dir, image_id)
    if tuple(report["image_size"]) != original.size:
        raise ValueError("Report coordinates and source-image dimensions do not match.")
    original.save(output / f"{image_id}_original.jpg", quality=95)
    all_boxes = original.copy()
    comparison = original.copy()
    truth_image = original.copy()
    mark_box(truth_image, truth, "GROUND TRUTH", "#00ff55")
    truth_image.save(output / f"{image_id}_ground_truth.jpg", quality=95)
    candidates: list[CandidateInspection] = []
    for index, proposal in enumerate(report["proposals"], 1):
        color = COLORS[(index - 1) % len(COLORS)]
        bbox = proposal["bbox"]
        if not (
            0 <= bbox[0] < bbox[2] <= original.width
            and 0 <= bbox[1] < bbox[3] <= original.height
        ):
            raise ValueError(f"Candidate {index} is outside the original image.")
        overlap = box_iou(bbox, truth)
        mark_box(
            all_boxes, bbox, f"P{index}", color, label_offset=((index - 1) % 3) * 45
        )
        mark_box(
            comparison, bbox, f"P{index}", color, label_offset=((index - 1) % 3) * 45
        )
        full_name = f"full_images/{image_id}_candidate_{index:03d}.jpg"
        full = original.copy()
        mark_box(full, bbox, f"P{index} | {model_status(proposal)}", color)
        full.save(output / full_name, quality=95)
        if (
            proposal.get("scoring_scope") == "region"
            and "scoring_input_image_path" in proposal
        ):
            card = _region_card(
                run / proposal["scoring_input_image_path"],
                proposal,
                index,
                overlap,
                image_id,
            )
        elif "evidence_image_path" in proposal:
            card = _uploaded_card(
                run / proposal["evidence_image_path"],
                proposal,
                index,
                color,
                overlap,
                image_id,
            )
        else:
            card = _legacy_card(
                original, run, proposal, index, color, overlap, image_id
            )
        card_name = f"candidate_views/{image_id}_candidate_{index:03d}.jpg"
        card.save(output / card_name, quality=95)
        candidates.append(
            {
                "candidate_id": index,
                "bbox": bbox,
                "sam3_score": proposal["sam3_score"],
                "verifier_score": proposal["verifier"]["score"],
                "verifier_answer": proposal["verifier"]["answer"],
                "verifier_reason": proposal["verifier"]["reason"],
                "candidate_score": proposal["score"],
                "target_iou": overlap,
                "full_image": full_name,
                "inspection_image": card_name,
            }
        )
        if "candidate_rank" in proposal:
            candidates[-1]["candidate_rank"] = proposal["candidate_rank"]
        if "candidate_verified" in proposal:
            candidates[-1]["candidate_verified"] = proposal["candidate_verified"]
        if (
            proposal.get("scoring_scope") == "region"
            and "scoring_input_image_path" in proposal
        ):
            candidates[-1]["model_input_image"] = os.path.relpath(
                run / proposal["scoring_input_image_path"], output
            )
            candidates[-1]["scoring_scope"] = "region"
            candidates[-1]["region_score"] = proposal["score"]
        elif "evidence_image_path" in proposal:
            candidates[-1]["model_input_image"] = os.path.relpath(
                run / proposal["evidence_image_path"], output
            )
            candidates[-1]["scoring_scope"] = "proposal"
    mark_box(comparison, truth, "GROUND TRUTH", "#00ff55", label_offset=145)
    all_boxes.save(output / f"{image_id}_all_predictions.jpg", quality=95)
    comparison.save(output / f"{image_id}_predictions_vs_ground_truth.jpg", quality=95)
    best = max(candidates, key=lambda row: row["target_iou"], default=None)
    summary: InspectionSummary = {
        "image_id": image_id,
        "question_id": question_id,
        "question": sample["question"],
        "annotated_answer": sample["answer"],
        "target_bbox": truth,
        "stop_reason": report["stop_reason"],
        "api_calls": report["usage"]["api_calls"],
        "elapsed_seconds": report["elapsed_seconds"],
        "reported_cost_usd": report["usage"]["reported_cost_usd"],
        "best_candidate_id": best["candidate_id"] if best else 0,
        "best_target_iou": best["target_iou"] if best else 0.0,
        "candidates": candidates,
    }
    (output / "inspection.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures = "\n".join(
        f'<figure><a href="{row["inspection_image"]}"><img src="{row["inspection_image"]}" loading="lazy"></a>'
        f"<figcaption>P{row['candidate_id']} | IoU {row['target_iou']:.3f} | "
        f"Candidate {row['candidate_score']:.3f} | "
        f"Qwen {100 * row['verifier_score']:.0f}/100 ({html.escape(row['verifier_answer'])}) | "
        f'<a href="{row["full_image"]}">Full image</a>'
        + (
            f' | <a href="{html.escape(row["model_input_image"], quote=True)}">Exact model input</a>'
            if "model_input_image" in row
            else ""
        )
        + "</figcaption></figure>"
        for row in candidates
    )
    page = f'''<!doctype html><html><meta charset="utf-8"><title>{image_id} result inspection</title>
<style>body{{background:#111827;color:#f1f5f9;font:17px system-ui;max-width:1500px;margin:32px auto;padding:20px}}a{{color:#67e8f9}}img{{max-width:100%}}figure{{margin:25px 0;border:1px solid #475569}}figcaption{{padding:12px}}</style>
<h1>Image {image_id}: bounding-box inspection</h1><p>{html.escape(summary["question"])}</p>
<p>Annotated answer: {html.escape(summary["annotated_answer"])}. Annotation boxes and answer are used only for this post-run inspection.</p>
<p>{summary["api_calls"]} HTTP calls | {summary["elapsed_seconds"]:.1f}s | reported cost ${summary["reported_cost_usd"]:.5f} | stop: {html.escape(summary["stop_reason"])}</p>
<p>Best overlap: P{summary["best_candidate_id"]}, IoU {summary["best_target_iou"]:.3f}.</p>
<h2>All predictions and ground truth</h2><a href="{image_id}_predictions_vs_ground_truth.jpg"><img src="{image_id}_predictions_vs_ground_truth.jpg"></a>
<p>Green marks the annotated target. Numbered colors mark predicted candidates. Open a candidate below to inspect the exact visual input sent to the model.</p>{figures}</html>'''
    (output / "index.html").write_text(page)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_new"))
    parser.add_argument("--image-id", default="0063")
    parser.add_argument("--question-id", default="0063_q01")
    args = parser.parse_args()
    summary = make_inspection(
        args.run_dir,
        args.output_dir,
        data_dir=args.data_dir,
        image_id=args.image_id,
        question_id=args.question_id,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "candidates"},
            indent=2,
        )
    )
    print(
        f"Annotated candidates: {len(summary['candidates'])}; open {args.output_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()
