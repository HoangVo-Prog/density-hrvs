"""Resumable full-dataset grounding runs; annotations are evaluation-only."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
from pathlib import Path
import threading
import torch
from collections.abc import Callable
from typing import NotRequired, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..core.contracts import CandidatePrompt, SAM3Detection, VerifierScore
from ..core.data import load_dataset_image
from ..core.dataset_manifest import load_inference_questions
from ..reporting.inspection import make_inspection
from ..models.qwen_scoring import load_qwen_scorer
from ..core.query import question_target
from ..models.scoring import ScoringView, _ignore_request
from ..models.cached_features import CachedFullImageFeatures
from ..models.regions import exec_dinov3_pca_on_image, load_dinov3
from ..search.runner import run_visual_search, save_search_result
from ..models.residency import staged_callable, staged_scorer
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
    elapsed_seconds: NotRequired[float]
    physical_generations: NotRequired[int]
    search_generations: NotRequired[int]
    primary_detector_calls: NotRequired[int]
    feature_encoder_calls: NotRequired[int]
    peak_gpu_allocated_bytes: NotRequired[int]
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
    parser.add_argument("--config-file", type=Path, help="JSON overrides applied after CLI defaults")
    parser.add_argument("--question-ids", nargs="+", help="Explicit question IDs, filtered before limits")
    parser.add_argument("--no-stages", action="store_true", help="Skip heavy per-node diagnostic images")
    args = parser.parse_args()
    if args.compact_existing:
        for checkpoint in sorted(args.output_dir.glob("*/completion.json")):
            compact_gallery(checkpoint.parent)
            print(f"Compacted {checkpoint.parent.name}", flush=True)
        return
    if args.workers < 1:
        parser.error("workers must be positive")
    rows = load_inference_questions(args.data_dir)
    questions = [
        Question(
            image_id=str(r["image_id"]),
            question_id=r["question_id"],
            question=r["question"],
        )
        for r in rows
    ]
    if args.question_ids:
        wanted = set(args.question_ids)
        questions = [q for q in questions if q["question_id"] in wanted]
        if {q["question_id"] for q in questions} != wanted:
            raise ValueError("Some requested question IDs are absent from the dataset")
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
    settings["region_fallback_limit"] = args.region_fallback_limit
    settings["discovery_mode"] = args.discovery_mode
    settings["discovery_tiles"] = args.discovery_tiles
    settings["discovery_candidates"] = args.discovery_candidates
    settings["discovery_seeds"] = args.discovery_seeds
    settings["discovery_overlap"] = args.discovery_overlap
    settings["answer_question"] = not args.no_final_answer
    if args.config_file is not None:
        overrides = json.loads(args.config_file.read_text())
        if not isinstance(overrides, dict):
            raise ValueError("config-file must contain an object")
        settings.update(overrides)
    from ..search.config import validate_search_config
    validate_search_config(settings)
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
    staging = settings.get("model_residency", "resident") == "staged"
    vision_device = "cpu" if staging else settings["device"]
    compute_device = "cuda" if settings["device"] == "auto" else settings["device"]
    dino = dino_processor = None
    if settings.get("partition_mode", "semantic") != "spatial" and not settings.get("feature_cache_root"):
        dino, dino_processor = load_dinov3(device=vision_device)
    qwen = load_qwen_scorer(settings["device"], model_id=settings["qwen_model"],
        revision=settings.get("qwen_revision"), cache_dir=settings.get("qwen_cache_dir"),
        quantization=settings.get("qwen_quantization", "none"))
    if staging:
        qwen = staged_scorer(qwen, compute_device)
    settings["scoring_backend"] = f"{qwen.model_id}@{qwen.model_revision}:{qwen.quantization}"
    llmdet = None
    if settings.get("primary_detector", "sam3") == "llmdet":
        from ..models.llmdet import LLMDetDetector
        llmdet = LLMDetDetector(device=compute_device, cache_dir=settings.get("llmdet_cache_dir"),
            threshold=settings.get("llmdet_threshold", .25), limit=settings["max_proposals_per_node"], staged=staging)
    else:
        sam, sam_processor = load_sam3_model(vision_device)
    gpu_lock = threading.Lock()

    def features(pixels: NDArray) -> NDArray[np.float32]:
        nonlocal dino, dino_processor
        with gpu_lock:
            if dino is None:
                dino, dino_processor = load_dinov3(device=vision_device)
            def encode():
                return exec_dinov3_pca_on_image(pixels, dino, dino_processor, max_side=settings["feature_max_side"])
            return staged_callable(encode, dino.to, compute_device)() if staging else encode()

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

    for prompt_method in ("locate_statement", "rewrite", "resolve", "resolve_structured", "compare_candidates", "answer", "answer_evidence"):
        method = getattr(qwen, prompt_method, None)
        if callable(method):
            setattr(score, prompt_method, guarded(method))

    for key in ("model_id", "model_revision", "quantization"):
        setattr(score, key, getattr(qwen, key))

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
            if llmdet is not None:
                return llmdet(pixels, concept)
            def proposal_call():
                return exec_sam3_on_image(pixels, sam, sam_processor, prompt=concept,
                    threshold=settings["sam3_threshold"], mask_threshold=settings["mask_threshold"],
                    max_detections=settings["max_proposals_per_node"])
            return staged_callable(proposal_call, sam.to, compute_device)() if staging else proposal_call()

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
            feature_provider = features
            if settings.get("feature_cache_root"):
                feature_provider = CachedFullImageFeatures(image,
                    Path(settings["feature_cache_root"]) / question["image_id"], features)
            config = settings.copy()
            config["feature_source"] = "cached_full_image" if settings.get("feature_cache_root") else "fresh_crop"
            config["sam_prompt"] = concept
            if args.workers == 1 and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            result = run_visual_search(
                image,
                query,
                question=question["question"],
                config=config,
                local_scorer=score,
                detector=detect,
                feature_extractor=feature_provider,
                artifact_dir=None if args.no_stages else folder / "stages",
                progress=lambda message: print(
                    f"{question['question_id']}: {message}", flush=True
                ),
            )
            outcome['elapsed_seconds'] = result['elapsed_seconds']
            for key in ('physical_generations','search_generations','primary_detector_calls','feature_encoder_calls'):
                outcome[key] = result['usage'].get(key,0)
            if args.workers == 1 and torch.cuda.is_available():
                outcome['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
            outcome["api_calls"] = result["usage"]["api_calls"]
            outcome["cost_usd"] = result["usage"]["reported_cost_usd"]
            outcome["local_score_calls"] = result["usage"].get("local_score_calls", 0)
            outcome["final_answer"] = result.get("final_answer", result.get("abstention", {})).get("answer", "")
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
    def completed_outcomes():
        if args.workers == 1:
            for question in questions:
                yield execute(question)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(execute, q) for q in questions]
                for future in as_completed(futures):
                    yield future.result()

    for outcome in completed_outcomes():
        outcomes.append(outcome)
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
