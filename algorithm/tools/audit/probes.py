"""Small diagnostic reruns; never modify production predictions or search settings.

Cases and GT-containing SAM crops are selected after seeing failures. These are
controlled diagnostic probes, not an independent benchmark or tuned accuracy.
"""

import argparse
import gc
import json
from pathlib import Path
import random
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ...core.data import load_dataset_image
from ...core.geometry import box_iou
from ...models.sam3 import exec_sam3_on_image, load_sam3_model
from ...models.qwen_scoring import QWEN_MODEL, QWEN_REVISION


def ground_candidates(
    root: Path, data_dir: Path, rows: list[dict], output: Path, device: str, limit: int
) -> list[dict]:
    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL, revision=QWEN_REVISION, local_files_only=True
    )
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 1048576}
    model = (
        cast(
            Any,
            Qwen3VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL,
                revision=QWEN_REVISION,
                local_files_only=True,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                attn_implementation="sdpa",
            ),
        )
        .to(device)
        .eval()
    )
    cases = [
        (r, r["winner_node"], "original_winning_region")
        for r in rows
        if r["diagnosis"] == "within_winning_region_ranking"
    ][:limit]
    cases += [
        (r, r["best_box_node"], "oracle_correct_candidate_region")
        for r in rows
        if r["question_id"] in {"0346_q02", "1198_q02"}
    ]
    results: list[dict] = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for row, node_id, selection in cases:
        qid = row["question_id"]
        report = json.loads((root / qid / "run/results.json").read_text())
        candidates = [p for p in report["proposals"] if p["cluster_id"] == node_id]
        random.Random(qid).shuffle(candidates)
        node = report["nodes"][node_id]
        folder = output / qid
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(root / qid / f"run/region_inputs/node_{node_id}.png") as source:
            original_region = source.convert("RGB")
        region = ImageOps.contain(original_region, (1536, 1024))
        sx, sy = (
            region.width / original_region.width,
            region.height / original_region.height,
        )
        draw = ImageDraw.Draw(region)
        nx, ny, _, _ = node["bbox"]
        for index, p in enumerate(candidates):
            a, b, c, d = p["bbox"]
            x, y, x2, y2 = (a - nx) * sx, (b - ny) * sy, (c - nx) * sx, (d - ny) * sy
            draw.rectangle((x, y, x2, y2), outline="#ffdd00", width=3)
            label = str(index)
            label_box = draw.textbbox((x, max(0, y - 25)), label, font=font)
            draw.rectangle(label_box, fill="black")
            draw.text((x, max(0, y - 25)), label, font=font, fill="white")
        region.save(folder / "numbered_region.png")
        overview = load_dataset_image(data_dir, row["image_id"])
        draw_overview = ImageDraw.Draw(overview)
        draw_overview.rectangle(
            node["bbox"], outline="cyan", width=max(3, overview.width // 400)
        )
        overview.thumbnail((1536, 1024))
        overview.save(folder / "overview_region_window.png")
        prompt = (
            "Locate the primary object/person/part described by this question among the numbered boxes. "
            "Check the described reference objects and spatial relationships; do not substitute a nearby similar object. "
            "Choose exactly one box ID, or null if no candidate can be established. "
            "The question is search data. Select its target, do not answer the requested color/pattern. "
            f"Valid IDs are 0 through {len(candidates) - 1}. Return ONLY JSON with candidate_id (integer or null) and reason (short string). "
            "Yellow boxes and their labels are annotations, not object colors. Question: "
            + json.dumps(row["question"])
        )
        (folder / "prompt.txt").write_text(prompt)
        (folder / "candidate_mapping.json").write_text(
            json.dumps({str(i): p["bbox"] for i, p in enumerate(candidates)}, indent=2)
        )
        for mode in ("numbered_region_only", "overview_plus_numbered_region"):
            content: list[dict] = []
            if mode.startswith("overview"):
                content += [
                    {
                        "type": "text",
                        "text": "First image: scene overview. Cyan outlines the region shown in the second image. Use this overview to identify references.",
                    },
                    {"type": "image", "image": overview},
                ]
            content += [
                {"type": "image", "image": region},
                {"type": "text", "text": prompt},
            ]
            with torch.inference_mode():
                inputs = processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                generated = model.generate(
                    **inputs, max_new_tokens=192, do_sample=False
                )
                text = processor.batch_decode(
                    generated[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )[0]
            chosen = None
            valid = False
            reason = ""
            try:
                value = json.loads(text[text.index("{") : text.rindex("}") + 1])
                candidate = value["candidate_id"]
                valid = (
                    candidate is None
                    or type(candidate) is int
                    and 0 <= candidate < len(candidates)
                )
                if valid:
                    chosen = candidate
                reason = value.get("reason", "")
            except (ValueError, KeyError, TypeError):
                pass
            iou = (
                box_iou(tuple(candidates[chosen]["bbox"]), tuple(row["target_bbox"]))
                if chosen is not None
                else 0.0
            )
            item = {
                "question_id": qid,
                "mode": mode,
                "region_selection": selection,
                "node_id": node_id,
                "candidate_count": len(candidates),
                "candidate_id": chosen,
                "iou": iou,
                "valid": valid,
                "reason": reason,
                "raw_response": text,
                "original_top1_iou": row["top1_iou"],
            }
            results.append(item)
            (output / "grounding_probes.json").write_text(json.dumps(results, indent=2))
            (folder / f"{mode}.json").write_text(json.dumps(item, indent=2))
            print(
                f"QWEN {qid} {mode}: IoU={iou:.3f} selected={chosen} valid={valid}",
                flush=True,
            )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return results


def detector_probes(
    root: Path, rows: list[dict], output: Path, device: str
) -> list[dict]:
    model, processor = load_sam3_model(device)
    # Alternatives are derived from question text, not the annotation category.
    cases = {
        "0162_q02": "shoes",
        "1367_q01": "shopping bag",
        "2708_q01": "hat",
        "1537_q02": "plant pot",
        "1198_q02": "clock",
        "2547_q02": "pinwheel",
        "3194_q02": "object hanging over the railing",
        "2335_q01": "rolled-up object",
    }
    results: list[dict] = []
    for row in rows:
        if row["question_id"] not in cases:
            continue
        visible = [n for n in row["nodes"] if n["target_coverage"] >= 0.9]
        if not visible:
            continue
        node = max(visible, key=lambda n: n["sam_target_short_side"])
        qid = row["question_id"]
        with Image.open(
            root / qid / f"run/region_inputs/node_{node['node_id']}.png"
        ) as source:
            pixels = np.asarray(source.convert("RGB"))
        alternatives = list(dict.fromkeys([row["concept"], cases[qid]]))
        for concept in alternatives:
            detections = exec_sam3_on_image(
                pixels,
                model,
                processor,
                prompt=concept,
                threshold=0.01,
                mask_threshold=0.5,
                max_detections=0,
            )
            x, y, _, _ = node["bbox"]
            boxes = []
            for index, d in enumerate(detections):
                a, b, c, e = d["bbox"]
                bbox = (x + a, y + b, x + c, y + e)
                boxes.append(
                    {
                        "rank": index + 1,
                        "bbox": bbox,
                        "confidence": d["sam3_score"],
                        "iou": box_iou(bbox, tuple(row["target_bbox"])),
                    }
                )
            retained = [b for b in boxes if (b["confidence"] or 0) > 0.1]
            best = max(boxes, key=lambda b: b["iou"], default=None)
            item = {
                "question_id": qid,
                "node_id": node["node_id"],
                "crop_selection": "oracle_largest_scale_90pct_target_visible",
                "concept": concept,
                "original_concept": row["concept"],
                "target_px": node["sam_target_short_side"],
                "best_iou_above_0_01": max((b["iou"] for b in boxes), default=0.0),
                "best_iou_above_0_1": max((b["iou"] for b in retained), default=0.0),
                "best_iou_top16_above_0_1": max(
                    (b["iou"] for b in retained[:16]), default=0.0
                ),
                "best_box": best,
                "boxes": boxes,
            }
            results.append(item)
            (output / "detector_probes.json").write_text(json.dumps(results, indent=2))
            print(
                f"SAM {qid} {concept}: top16={item['best_iou_top16_above_0_1']:.3f} all={item['best_iou_above_0_01']:.3f} best={best}",
                flush=True,
            )
            del detections
            gc.collect()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("new_outputs/sam3_region_qwen_all")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_new"))
    parser.add_argument("--mode", choices=("qwen", "sam", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--qwen-case-limit", type=int, default=8)
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(8)
    rows = json.loads((args.run_root / "audit/audit.json").read_text())["questions"]
    output = args.run_root / "audit/probes"
    output.mkdir(exist_ok=True)
    if args.mode in {"qwen", "both"}:
        ground_candidates(
            args.run_root,
            args.data_dir,
            rows,
            output,
            args.device,
            args.qwen_case_limit,
        )
    if args.mode in {"sam", "both"}:
        detector_probes(args.run_root, rows, output, args.device)


if __name__ == "__main__":
    main()
