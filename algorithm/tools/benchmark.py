"""Resumable full-dataset grounding runs; annotations are evaluation-only."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
from pathlib import Path
import threading
from collections.abc import Callable
from typing import NotRequired, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..core.contracts import CandidatePrompt, SAM3Detection, VerifierScore
from ..core.data import load_dataset_image
from ..reporting.inspection import make_inspection
from ..models.qwen_scoring import load_qwen_scorer
from ..core.query import question_target
from ..models.scoring import ScoringView, _ignore_request
from ..models.regions import exec_dinov3_pca_on_image, load_dinov3
from ..search.runner import run_visual_search, save_search_result
from ..models.sam3 import exec_sam3_on_image, load_sam3_model
from ..search.config import default_search_config


class Question(TypedDict):
    """Only inference inputs copied from the annotation manifest."""

    image_id: str  # Source image identifier, resolved locally or inside the ZIP.
    question_id: str  # Stable checkpoint/output folder name.
    question: str  # Natural-language question; answers and boxes are excluded.


class Outcome(TypedDict):
    """Per-question completion record used to resume and summarize the batch."""

    question_id: str  # Identifier of the attempted question.
    status: str  # complete or error; errors remain eligible for another attempt.
    api_calls: int  # Physical search HTTP attempts, including retries.
    cost_usd: float  # Provider-reported cost, when returned.
    error: str  # Search or export failure; empty for successful runs.
    local_score_calls: int  # Local Qwen generations, separate from HTTP attempts.
    top1_iou: float  # Highest-ranked proposal overlap, evaluated after inference.
    best_iou: float  # Best overlap anywhere in the candidate list.
    warnings: NotRequired[
        list[str]
    ]  # Skipped malformed verifier regions, not hidden as zero scores.


def compact_gallery(folder: Path) -> None:
    """Bound display-copy sizes; preserve raw proposals and exact verifier PNGs."""
    paths = list((folder / "run" / "clusters").glob("*.png"))
    paths += list((folder / "inspection").rglob("*.jpg"))
    paths += [folder / "run" / "preview.jpg"]
    for path in paths:
        with Image.open(path) as source:
            image = source.copy()
        side = (
            1024
            if path.parent.name in {"clusters", "full_images"}
            or path.name.endswith(("_original.jpg", "_ground_truth.jpg"))
            else 1536
        )
        image.thumbnail((side, side))
        if path.suffix == ".jpg":
            image.save(path, quality=85, optimize=True)
        else:
            image.save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_new"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("new_outputs/sam3_region_qwen_10")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-api-calls", type=int, default=32)
    parser.add_argument(
        "--limit-images",
        type=int,
        default=10,
        help="First N image IDs in sorted order; zero selects all",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="First N sorted questions after image selection; zero selects all",
    )
    parser.add_argument("--max-local-scores", type=int, default=32)
    parser.add_argument(
        "--verification-mode",
        choices=("region", "candidate"),
        default="region",
        help="Verify the whole cluster with numbered top boxes, or one sheet per candidate",
    )
    parser.add_argument(
        "--region-candidate-limit",
        type=int,
        default=5,
        help="Top SAM3 boxes drawn for whole-region verification; 0 disables",
    )
    parser.add_argument(
        "--compact-existing",
        action="store_true",
        help="Resize existing display copies without running inference",
    )
    args = parser.parse_args()
    if args.compact_existing:
        for checkpoint in sorted(args.output_dir.glob("*/completion.json")):
            compact_gallery(checkpoint.parent)
            print(f"Compacted {checkpoint.parent.name}", flush=True)
        return
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest = sorted(args.data_dir.glob("hrvs_bbox_samples_*.json"))[-1]
    rows = json.loads(manifest.read_text())
    questions = [
        Question(
            image_id=str(r["image_id"]),
            question_id=r["question_id"],
            question=r["question"],
        )
        for r in rows
    ]
    if args.limit_images < 0:
        parser.error("limit-images must be nonnegative")
    image_ids = sorted({q["image_id"] for q in questions})
    selected_ids = set(
        image_ids[: args.limit_images] if args.limit_images else image_ids
    )
    questions = sorted(
        (q for q in questions if q["image_id"] in selected_ids),
        key=lambda q: q["question_id"],
    )
    if args.limit_questions < 0:
        parser.error("limit-questions must be nonnegative")
    if args.limit_questions:
        questions = questions[: args.limit_questions]
    selected_ids = {q["image_id"] for q in questions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(json.dumps(questions, indent=2))
    settings = default_search_config()
    settings["max_api_calls"] = args.max_api_calls
    settings["max_local_scores"] = args.max_local_scores
    settings["verification_mode"] = args.verification_mode
    settings["region_candidate_limit"] = args.region_candidate_limit
    settings_path = args.output_dir / "config.json"
    serialized = json.loads(json.dumps(settings))
    if settings_path.exists() and json.loads(settings_path.read_text()) != serialized:
        raise ValueError(
            "Existing batch uses different settings; choose another output directory"
        )
    settings_path.write_text(json.dumps(settings, indent=2))
    print(
        f"Loading shared models for {len(questions)} questions; workers={args.workers}",
        flush=True,
    )
    dino, dino_processor = load_dinov3()
    qwen = load_qwen_scorer()
    sam, sam_processor = load_sam3_model()
    gpu_lock = threading.Lock()

    def features(pixels: NDArray) -> NDArray[np.float32]:
        with gpu_lock:
            return exec_dinov3_pca_on_image(
                pixels, dino, dino_processor, max_side=settings["feature_max_side"]
            )

    def score(
        image: Image.Image,
        query: str,
        *,
        view_type: ScoringView = "region",
        on_request: Callable[[], None] = _ignore_request,
    ) -> VerifierScore:
        with gpu_lock:
            return qwen(image, query, view_type=view_type, on_request=on_request)

    def guarded(method):
        def call(*args, **kwargs):
            with gpu_lock:
                return method(*args, **kwargs)

        return call

    for prompt_method in ("rewrite", "resolve"):
        method = getattr(qwen, prompt_method, None)
        if callable(method):
            setattr(score, prompt_method, guarded(method))

    def score_candidates(
        image: Image.Image,
        query: str,
        candidates: list[CandidatePrompt],
        *,
        on_request: Callable[[], None] = _ignore_request,
    ) -> dict[int, VerifierScore]:
        with gpu_lock:
            return qwen.score_candidates(
                image, query, candidates, on_request=on_request
            )

    setattr(score, "score_candidates", score_candidates)

    def detect(pixels: NDArray, concept: str) -> list[SAM3Detection]:
        with gpu_lock:
            return exec_sam3_on_image(
                pixels,
                sam,
                sam_processor,
                prompt=concept,
                threshold=settings["sam3_threshold"],
                mask_threshold=settings["mask_threshold"],
                max_detections=settings["max_proposals_per_node"],
            )

    def execute(question: Question) -> Outcome:
        folder = args.output_dir / question["question_id"]
        checkpoint = folder / "completion.json"
        query, concept = question_target(question["question"])
        if checkpoint.exists():
            previous = cast(Outcome, json.loads(checkpoint.read_text()))
            report_path = folder / "run" / "results.json"
            if previous["status"] == "complete" and report_path.exists():
                report = json.loads(report_path.read_text())
                if report["query"] == query:
                    compact_gallery(folder)
                    return previous
        folder.mkdir(parents=True, exist_ok=True)
        outcome: Outcome = {
            "question_id": question["question_id"],
            "status": "error",
            "api_calls": 0,
            "cost_usd": 0.0,
            "error": "",
            "local_score_calls": 0,
            "top1_iou": 0.0,
            "best_iou": 0.0,
        }
        try:
            image = np.asarray(load_dataset_image(args.data_dir, question["image_id"]))
            config = settings.copy()
            config["sam_prompt"] = concept
            result = run_visual_search(
                image,
                query,
                config=config,
                local_scorer=score,
                detector=detect,
                feature_extractor=features,
                artifact_dir=folder / "stages",
                progress=lambda message: print(
                    f"{question['question_id']}: {message}", flush=True
                ),
            )
            outcome["api_calls"] = result["usage"]["api_calls"]
            outcome["cost_usd"] = result["usage"]["reported_cost_usd"]
            outcome["local_score_calls"] = result["usage"].get("local_score_calls", 0)
            outcome["warnings"] = result.get("warnings", [])
            save_search_result(result, image, folder / "run")
            inspection = make_inspection(
                folder / "run",
                folder / "inspection",
                data_dir=args.data_dir,
                image_id=question["image_id"],
                question_id=question["question_id"],
            )
            outcome["top1_iou"] = (
                inspection["candidates"][0]["target_iou"]
                if inspection["candidates"]
                else 0.0
            )
            outcome["best_iou"] = inspection["best_target_iou"]
            compact_gallery(folder)
            outcome["error"] = result["error"]
            outcome["status"] = "error" if result["error"] else "complete"
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(outcome, indent=2))
        temporary.replace(checkpoint)
        return outcome

    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, q) for q in questions]
        for future in as_completed(futures):
            outcomes.append(future.result())
            summary = {
                "total": len(questions),
                "attempted": len(outcomes),
                "completed": sum(o["status"] == "complete" for o in outcomes),
                "api_calls": sum(o["api_calls"] for o in outcomes),
                "reported_cost_usd": sum(o["cost_usd"] for o in outcomes),
                "outcomes": sorted(outcomes, key=lambda o: o["question_id"]),
            }
            summary.update(
                images=len(selected_ids),
                local_score_calls=sum(o["local_score_calls"] for o in outcomes),
                top1_iou_at_50=sum(o["top1_iou"] >= 0.5 for o in outcomes),
                any_iou_at_50=sum(o["best_iou"] >= 0.5 for o in outcomes),
            )
            summary["questions_with_recoverable_errors"] = sum(
                bool(o.get("warnings")) for o in outcomes
            )
            temporary = args.output_dir / "summary.tmp"
            temporary.write_text(json.dumps(summary, indent=2))
            temporary.replace(args.output_dir / "summary.json")
            links = "\n".join(
                f'<li><a href="{html.escape(o["question_id"])}/inspection/index.html">{html.escape(o["question_id"])}</a> | <a href="{html.escape(o["question_id"])}/stages/tree/index.html">Tree and stages</a>: {o["status"]} {html.escape(o["error"])}</li>'
                for o in summary["outcomes"]
            )
            (args.output_dir / "index.html").write_text(
                f'<html><meta charset="utf-8"><h1>{html.escape(args.data_dir.name)}: {summary["completed"]}/{len(questions)} complete</h1><ul>{links}</ul></html>'
            )
            print(f"BATCH {len(outcomes)}/{len(questions)}: {outcomes[-1]}", flush=True)


if __name__ == "__main__":
    main()
