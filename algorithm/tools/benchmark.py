"""Resumable full-dataset grounding runs; annotations are evaluation-only."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import hashlib
import json
import shutil
from pathlib import Path
import threading
from collections.abc import Callable
from typing import TypedDict, cast

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from ..core.contracts import CandidatePrompt, SAM3Detection, VerifierScore
from ..core.data import load_dataset_image
from ..search.region_scoring import bbox_iou
from ..models.qwen_scoring import QWEN_MODEL, QWEN_REVISION, load_qwen_scorer
from ..models.vllm_scoring import DEFAULT_ENDPOINT, VLLM_MODELS, VllmScorer
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
    final_answer: NotRequired[str]  # Qwen text answer from the top-ranked region.
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


def inspect_result(result, target, folder):
    box = (target["x"], target["y"], target["x"] + target["width"], target["y"] + target["height"]) if target else None
    candidates = [{"bbox": p["bbox"], "score": p["score"],
                   "target_iou": bbox_iou(tuple(p["bbox"]), box) if box else None}
                  for p in result["ranked_regions"]]
    record = {"target_bbox": box, "candidates": candidates,
              "best_target_iou": max((c["target_iou"] for c in candidates), default=0.0) if box else None}
    folder.mkdir(exist_ok=True)
    (folder / "inspection.json").write_text(json.dumps(record, indent=2))
    (folder / "index.html").write_text('<a href="inspection.json">Candidate boxes and post-hoc IoU</a>')
    return record


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
        "--region-fallback-limit",
        type=int,
        default=3,
        help="Ambiguous regions re-check this many candidates with detail sheets",
    )
    parser.add_argument(
        "--discovery-mode",
        choices=("off", "sam3_qwen"),
        default="off",
        help="Seed the tree from tiled SAM3 candidates reranked by Qwen",
    )
    parser.add_argument("--discovery-tiles", type=int, default=3)
    parser.add_argument("--discovery-candidates", type=int, default=5)
    parser.add_argument("--discovery-seeds", type=int, default=3)
    parser.add_argument("--discovery-overlap", type=float, default=0.10)
    parser.add_argument(
        "--no-final-answer",
        action="store_true",
        help="Skip the post-search Qwen answer from the top-ranked region",
    )
    parser.add_argument(
        "--compact-existing",
        action="store_true",
        help="Resize existing display copies without running inference",
    )
    parser.add_argument("--vlm-backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--vlm-model", choices=tuple(VLLM_MODELS), default=QWEN_MODEL)
    parser.add_argument("--vlm-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--lean-artifacts", action="store_true", help="Keep all JSON but only a few key images per question")
    args = parser.parse_args()
    if args.compact_existing:
        for checkpoint in sorted(args.output_dir.glob("*/completion.json")):
            compact_gallery(checkpoint.parent)
            print(f"Compacted {checkpoint.parent.name}", flush=True)
        return
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.vlm_backend == "transformers" and args.vlm_model != QWEN_MODEL:
        parser.error("the transformers backend only loads " + QWEN_MODEL)
    manifest = sorted(args.data_dir.glob("hrvs_bbox_samples_*.json"))[-1]
    rows = json.loads(manifest.read_text())
    targets = {r["question_id"]: r.get("target_bbox") for r in rows}
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
    provenance = {
        "dataset_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("algorithm").rglob("*.py")},
    }
    vlm = {"backend": args.vlm_backend, "model": args.vlm_model}
    if args.vlm_backend == "vllm":
        vlm.update(revision=VLLM_MODELS[args.vlm_model]["revision"], endpoint=args.vlm_endpoint,
                   processor=VLLM_MODELS[args.vlm_model]["processor"])
    else:
        vlm.update(revision=QWEN_REVISION, processor=VLLM_MODELS[QWEN_MODEL]["processor"])
    provenance["vlm"] = vlm
    if args.lean_artifacts:
        provenance["artifacts"] = "lean"
    provenance_path = args.output_dir / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Dataset or implementation changed; use a fresh output directory")
    provenance_path.write_text(json.dumps(provenance, indent=2))
    (args.output_dir / "selection.json").write_text(json.dumps(questions, indent=2))
    settings = default_search_config()
    settings["max_api_calls"] = args.max_api_calls
    settings["max_local_scores"] = args.max_local_scores
    settings["verification_mode"] = args.verification_mode
    settings["region_candidate_limit"] = args.region_candidate_limit
    settings["region_fallback_limit"] = args.region_fallback_limit
    settings["discovery_mode"] = args.discovery_mode
    settings["discovery_tiles"] = args.discovery_tiles
    settings["discovery_candidates"] = args.discovery_candidates
    settings["discovery_seeds"] = args.discovery_seeds
    settings["discovery_overlap"] = args.discovery_overlap
    settings["answer_question"] = not args.no_final_answer
    settings["scoring_backend"] += ":pyes"
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
    free, total = torch.cuda.mem_get_info()
    if free < 16 * 1024**3:
        raise RuntimeError("Need at least 16 GiB free VRAM")
    torch.cuda.set_per_process_memory_fraction(14 * 1024**3 / total)
    dino, dino_processor = load_dinov3()
    qwen = VllmScorer(args.vlm_model, args.vlm_endpoint) if args.vlm_backend == "vllm" else load_qwen_scorer()
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

    for prompt_method in ("rewrite", "resolve", "answer"):
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
    if args.vlm_backend == "vllm":
        score = qwen
    else:
        score.model_id, score.model_revision, score.backend = QWEN_MODEL, QWEN_REVISION, "transformers"

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
        if args.lean_artifacts and shutil.disk_usage(args.output_dir).free < 16 * 1024**3:
            raise RuntimeError("Stopping lean run to preserve 16 GiB free disk space")
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
                question=question["question"],
                config=config,
                local_scorer=score,
                detector=detect,
                feature_extractor=features,
                artifact_dir=None if args.lean_artifacts else folder / "stages",
                progress=lambda message: print(
                    f"{question['question_id']}: {message}", flush=True
                ),
            )
            outcome["api_calls"] = result["usage"]["api_calls"]
            outcome["cost_usd"] = result["usage"]["reported_cost_usd"]
            outcome["local_score_calls"] = result["usage"].get("local_score_calls", 0)
            outcome["final_answer"] = result.get("final_answer", {}).get("answer", "")
            outcome["warnings"] = result.get("warnings", [])
            save_search_result(result, image, folder / "run", lean=args.lean_artifacts)
            inspection = inspect_result(result, targets[question["question_id"]], folder / "inspection")
            outcome["localization_available"] = inspection["target_bbox"] is not None
            outcome["top1_iou"] = (inspection["candidates"][0]["target_iou"] or 0.0) if inspection["candidates"] else 0.0
            outcome["best_iou"] = inspection["best_target_iou"] or 0.0
            outcome["top10_hit"] = any((c["target_iou"] or 0.0) >= 0.5 for c in inspection["candidates"][:10])
            compact_gallery(folder)
            outcome["error"] = result["error"]
            outcome["status"] = "error" if result["error"] else "complete"
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(outcome, indent=2))
        temporary.replace(checkpoint)
        return outcome

    def tree_link(o: Outcome) -> str:
        if args.lean_artifacts:
            return ""
        return f' | <a href="{html.escape(o["question_id"])}/stages/tree/index.html">Tree and stages</a>'

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
            summary["localization_evaluated"] = sum(o.get("localization_available", False) for o in outcomes)
            summary["top10_iou_at_50"] = sum(o.get("top10_hit", False) for o in outcomes)
            summary["vlm"] = vlm
            if args.vlm_backend == "vllm":
                summary["vllm_http_requests"] = qwen.http_requests
                summary["logical_probability_evals"] = qwen.probability_forwards
                summary["probability_note"] = "Candidate evaluations; server batching means these are not physical forwards."
            else:
                summary["physical_probability_forwards"] = qwen.probability_forwards
            summary["questions_with_recoverable_errors"] = sum(
                bool(o.get("warnings")) for o in outcomes
            )
            temporary = args.output_dir / "summary.tmp"
            temporary.write_text(json.dumps(summary, indent=2))
            temporary.replace(args.output_dir / "summary.json")
            links = "\n".join(
                f'<li><a href="{html.escape(o["question_id"])}/inspection/index.html">{html.escape(o["question_id"])}</a>{tree_link(o)}: {o["status"]} {html.escape(o["error"])}</li>'
                for o in summary["outcomes"]
            )
            (args.output_dir / "index.html").write_text(
                f'<html><meta charset="utf-8"><h1>{html.escape(args.data_dir.name)}: {summary["completed"]}/{len(questions)} complete</h1><ul>{links}</ul></html>'
            )
            print(f"BATCH {len(outcomes)}/{len(questions)}: {outcomes[-1]}", flush=True)


if __name__ == "__main__":
    main()
