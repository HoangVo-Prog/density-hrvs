"""One-shot SAM3 tiling + Qwen presence reranker feasibility probe."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ..core.data import load_dataset_image
from ..core.errors import InvalidScoreResponse
from ..core.geometry import box_iou
from ..core.query import question_target
from ..models.evidence import build_proposal_evidence
from ..models.qwen_scoring import load_qwen_scorer
from ..models.sam3 import exec_sam3_on_image, load_sam3_model
from ..search.config import default_search_config
from ..search.region_scoring import distinct_detections


def detection(bbox, score, prompt):
    return {
        "bbox": bbox,
        "score": score,
        "sam3_score": score,
        "mask": np.zeros((1, 1), dtype=bool),
        "image": np.zeros((1, 1, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def summarize(results):
    return {
        "images": len({r["image_id"] for r in results}),
        "questions": len(results),
        "pool_has_gt": sum(r["pool_best_iou"] >= 0.5 for r in results),
        "qwen_top1": sum(r["top_iou"]["top1"] for r in results),
        "qwen_top2": sum(r["top_iou"]["top2"] for r in results),
        "qwen_top3": sum(r["top_iou"]["top3"] for r in results),
        "center_top1": sum(r["top_center"]["top1"] for r in results),
        "center_top2": sum(r["top_center"]["top2"] for r in results),
        "center_top3": sum(r["top_center"]["top3"] for r in results),
        "sam3_top1": sum(r["sam3_top_iou"]["top1"] for r in results),
        "sam3_top2": sum(r["sam3_top_iou"]["top2"] for r in results),
        "sam3_top3": sum(r["sam3_top_iou"]["top3"] for r in results),
        "mean_pool_best_iou": float(np.mean([r["pool_best_iou"] for r in results]))
        if results
        else 0.0,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_official"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/rerank_probe.json")
    )
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--top-candidates", type=int, default=20)
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--overlap", type=float, default=0.10)
    return parser


def main():
    args = build_parser().parse_args()
    manifest = (
        args.manifest or sorted(args.data_dir.glob("hrvs_bbox_samples_*.json"))[-1]
    )
    rows = json.loads(manifest.read_text())
    if args.limit_questions:
        rows = rows[: args.limit_questions]
    config = default_search_config()
    qwen = load_qwen_scorer()
    sam, sam_processor = load_sam3_model()
    print(f"selected {len(rows)} questions", flush=True)

    results = []
    image_cache = {}
    start = time.time()
    for index, row in enumerate(rows, 1):
        image_id = str(row["image_id"])
        question = row["question"]
        query, concept = question_target(question)
        resolved = getattr(qwen, "resolve", None)
        prompt = resolved(query) if callable(resolved) else concept
        prompt = prompt or concept or query
        if image_id not in image_cache:
            image_cache[image_id] = load_dataset_image(args.data_dir, image_id).convert(
                "RGB"
            )
        source = image_cache[image_id]
        image = np.asarray(source)
        height, width = image.shape[:2]
        target = row["target_bbox"]
        gt = (
            float(target["x"]),
            float(target["y"]),
            float(target["x"] + target["width"]),
            float(target["y"] + target["height"]),
        )

        detections = []
        for gy in range(args.tiles):
            for gx in range(args.tiles):
                x0 = int(max(0, (gx / args.tiles - args.overlap) * width))
                x1 = int(min(width, ((gx + 1) / args.tiles + args.overlap) * width))
                y0 = int(max(0, (gy / args.tiles - args.overlap) * height))
                y1 = int(min(height, ((gy + 1) / args.tiles + args.overlap) * height))
                crop = image[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                found = exec_sam3_on_image(
                    crop,
                    sam,
                    sam_processor,
                    prompt=prompt,
                    threshold=config["sam3_threshold"],
                    mask_threshold=config["mask_threshold"],
                    max_detections=8,
                )
                for item in found:
                    a, b, c, d = item["bbox"]
                    bbox = (x0 + a, y0 + b, min(width, x0 + c), min(height, y0 + d))
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        detections.append(detection(bbox, item["sam3_score"], prompt))

        distinct = distinct_detections(detections, config)
        candidates = distinct[: args.top_candidates]
        ranked = []
        for number, candidate in enumerate(candidates, 1):
            try:
                view = build_proposal_evidence(
                    source,
                    candidate["bbox"],
                    max_side=config["verification_max_side"],
                    context_padding=config["context_padding"],
                    detail_padding=config["detail_padding"],
                )
            except ValueError:
                continue
            try:
                rating = qwen(
                    Image.fromarray(view["image"]),
                    query,
                    view_type="context_detail",
                )
            except InvalidScoreResponse:
                rating = {
                    "score": 0.0,
                    "answer": "Uncertain",
                    "reason": "invalid JSON after retry",
                    "model_id": "invalid",
                }
            ranked.append(
                {
                    "rank_sam3": number,
                    "bbox": candidate["bbox"],
                    "sam3_score": candidate["sam3_score"],
                    "qwen_score": rating["score"],
                    "qwen_answer": rating["answer"],
                    "iou": box_iou(candidate["bbox"], gt),
                }
            )
        ranked.sort(
            key=lambda item: (item["qwen_score"], item["sam3_score"]), reverse=True
        )
        pool_best = max((item["iou"] for item in ranked), default=0.0)
        top = {
            f"top{k}": any(item["iou"] >= 0.5 for item in ranked[:k]) for k in (1, 2, 3)
        }
        center = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        top_center = {
            f"top{k}": any(
                item["bbox"][0] <= center[0] <= item["bbox"][2]
                and item["bbox"][1] <= center[1] <= item["bbox"][3]
                for item in ranked[:k]
            )
            for k in (1, 2, 3)
        }
        sam3_top = {
            f"top{k}": any(
                item["iou"] >= 0.5
                for item in sorted(
                    ranked, key=lambda item: item["sam3_score"], reverse=True
                )[:k]
            )
            for k in (1, 2, 3)
        }
        result = {
            "image_id": image_id,
            "question_id": row["question_id"],
            "question": question,
            "prompt": prompt,
            "target_bbox": gt,
            "pool_size": len(ranked),
            "pool_best_iou": pool_best,
            "top_iou": top,
            "top_center": top_center,
            "sam3_top_iou": sam3_top,
            "ranked": ranked[:10],
        }
        results.append(result)
        args.output.write_text(
            json.dumps(
                {"partial": True, "summary": summarize(results), "results": results},
                indent=2,
            )
        )
        print(
            f"[{index}/{len(rows)}] {image_id} {row['question_id']} pool={len(ranked)} "
            f"best={pool_best:.3f} qwen_top1={top['top1']} top2={top['top2']} top3={top['top3']} "
            f"center_top1={top_center['top1']} center_top2={top_center['top2']} "
            f"center_top3={top_center['top3']} sam3_top1={sam3_top['top1']}",
            flush=True,
        )

    summary = summarize(results)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("saved", args.output, flush=True)
    print(f"elapsed {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
