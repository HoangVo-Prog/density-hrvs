"""Self-service image + question runner with live stages and decision logs.

Run from the project root: outputs/cvsearch_runtime/bin/python -m algorithm_refactored.tools.run
No arguments opens an interactive setup. All models run locally.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import traceback
from typing import TypedDict
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
import torch

from ..core.data import load_dataset_image
from ..core.events import EventData, create_event_logger
from ..reporting.inspection import make_inspection
from ..core.query import question_target
from ..reporting.run_viewer import RunRequest, RunStatus, render_run_page, utc_now
from ..search.runner import run_visual_search, save_search_result
from ..search.config import default_search_config, validate_search_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatasetQuestion(TypedDict):
    """Only question text and identifiers are passed from annotations to inference."""

    question_id: str
    image_id: str
    question: str


def dataset_question(data_dir: Path, question_id: str) -> DatasetQuestion:
    """Resolve an exact question ID without returning answer or target coordinates."""
    matches = []
    for path in sorted(data_dir.glob("hrvs_bbox_samples_*.json")):
        for row in json.loads(path.read_text()):
            if row.get("question_id") == question_id:
                matches.append(
                    DatasetQuestion(
                        question_id=question_id,
                        image_id=str(row["image_id"]),
                        question=row["question"],
                    )
                )
    if not matches:
        raise ValueError(f"Question ID {question_id!r} was not found in {data_dir}")
    if any(row != matches[0] for row in matches[1:]):
        raise ValueError(f"Question ID {question_id!r} is ambiguous")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image",
        type=Path,
        help="Your image file; EXIF orientation is applied before inference",
    )
    source.add_argument(
        "--image-id", help="Dataset image ID, e.g. 3308; images can stay inside the ZIP"
    )
    question = parser.add_mutually_exclusive_group()
    question.add_argument(
        "--question",
        "--query",
        dest="question",
        help="Your full question or target description",
    )
    question.add_argument(
        "--question-id",
        help="Dataset question ID, e.g. 3308_q01; also selects its image",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for the image, question and detector concept",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/data_new")
    parser.add_argument(
        "--detector-prompt",
        help="SAM3 object concept, e.g. wristband or clock; overrides automatic heuristic",
    )
    parser.add_argument(
        "--no-sam-prompt-rewrite",
        action="store_true",
        help="Keep the heuristic SAM3 concept; do not ask local Qwen for an object-first phrase",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh output directory; otherwise create a timestamped run under new_outputs/manual_runs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda or cpu; choose cpu when GPU memory is occupied",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="CPU inference threads when --device cpu is selected",
    )
    parser.add_argument(
        "--max-scores",
        type=int,
        default=32,
        help="Local Qwen generation budget, including repairs; 0 is unlimited",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=16,
        help="SAM3 proposals retained before NMS; 0 keeps all",
    )
    parser.add_argument(
        "--candidate-check-limit",
        type=int,
        default=2,
        help="Maximum SAM3 candidates verified per region; 0 disables verification",
    )
    parser.add_argument(
        "--verification-mode",
        choices=("region", "candidate"),
        default="region",
        help="Verify the whole cluster together or one evidence sheet per candidate",
    )
    parser.add_argument(
        "--region-candidate-limit",
        type=int,
        default=5,
        help="Top SAM3 boxes drawn for whole-region verification; 0 disables",
    )
    parser.add_argument(
        "--legacy-region-qwen",
        action="store_true",
        help="Score the whole region before detection instead of candidate sheets",
    )
    parser.add_argument(
        "--candidate-query-mode",
        choices=("full", "target_phrase"),
        default="full",
        help="Description shown with each candidate evidence sheet",
    )
    parser.add_argument(
        "--sam-threshold",
        type=float,
        default=0.1,
        help="SAM3 confidence threshold; raw filtered boxes are still logged",
    )
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument(
        "--strategy", choices=("agent-uct", "uct", "greedy"), default="agent-uct"
    )
    parser.add_argument("--exploration-weight", type=float, default=1.4)
    parser.add_argument("--cost-weight", type=float, default=0.2)
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Skip optional post-run ground-truth inspection for dataset question IDs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(arguments)
    interactive = args.interactive or not arguments
    if interactive:
        if not (args.image or args.image_id or args.question_id):
            value = input("Image file path or dataset image ID (e.g. 3308): ").strip()
            if value.isdigit():
                args.image_id = value.zfill(4)
            else:
                args.image = Path(value).expanduser()
        if not (args.question or args.question_id):
            value = input(
                "Question text or dataset question ID (e.g. 3308_q01): "
            ).strip()
            if re.fullmatch(r"\d+_q\d+", value):
                args.question_id = value
            else:
                args.question = value
    try:
        if args.question_id:
            if args.image is not None:
                raise ValueError(
                    "Use --question text for your own image; --question-id selects a dataset image"
                )
            sample = dataset_question(args.data_dir, args.question_id)
            if args.image_id and str(args.image_id).zfill(4) != sample[
                "image_id"
            ].zfill(4):
                raise ValueError(
                    "The requested image ID does not match the dataset question"
                )
            args.image_id = sample["image_id"]
            args.question = sample["question"]
        if not args.question or not (args.image or args.image_id):
            raise ValueError(
                "Provide an image and --question, use --question-id, or start with --interactive"
            )
        query, concept = question_target(args.question)
        if interactive and not args.detector_prompt:
            print(
                f"Automatic SAM3 concept: {concept!r}. Check that this names the object you want to detect."
            )
            args.detector_prompt = (
                input(f"SAM3 object concept [{concept}]: ").strip() or concept
            )
        concept = args.detector_prompt.strip() if args.detector_prompt else concept
        if not concept:
            raise ValueError("Provide a nonempty --detector-prompt")
        config = default_search_config()
        config.update(
            device=args.device,
            max_local_scores=args.max_scores,
            max_iterations=args.max_iterations,
            max_depth=args.max_depth,
            max_proposals_per_node=args.max_boxes,
            sam3_threshold=args.sam_threshold,
            sam3_first=not args.legacy_region_qwen,
            candidate_check_limit=args.candidate_check_limit,
            verification_mode=args.verification_mode,
            region_candidate_limit=args.region_candidate_limit,
            candidate_query_mode=args.candidate_query_mode,
            sam_prompt=concept,
            rewrite_sam_prompt=(
                not args.no_sam_prompt_rewrite and not args.detector_prompt
            ),
            region_nms_iou=args.nms_iou,
            strategy=args.strategy,
            exploration_weight=args.exploration_weight,
            cost_weight=args.cost_weight,
        )
        validate_search_config(config)
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        if args.device == "cpu":
            torch.set_num_threads(args.cpu_threads)
        if args.image_id:
            image = load_dataset_image(args.data_dir, args.image_id)
            image_source = f"dataset:{args.image_id} ({args.data_dir})"
        else:
            if args.image is None:
                raise ValueError("Provide an image file or a dataset image ID")
            image_path = args.image.expanduser().resolve()
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image_source = str(image_path)
        name = (
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
        )
        output = (
            (args.output_dir or PROJECT_ROOT / "new_outputs/manual_runs" / name)
            .expanduser()
            .absolute()
        )
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                f"Output directory is not empty: {output}. Choose a fresh folder; existing results are preserved."
            )
        output.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    request: RunRequest = {
        "question": args.question,
        "query": query,
        "detector_prompt": concept,
        "config": config,
        "image_source": image_source,
    }
    if args.image_id:
        request["image_id"] = str(args.image_id)
    if args.question_id:
        request["question_id"] = args.question_id
    status: RunStatus = {
        "state": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }
    (output / "request.json").write_text(json.dumps(request, indent=2))
    preview = image.copy()
    preview.thumbnail((1536, 1024))
    preview.save(output / "source_preview.jpg", quality=90)
    logger = create_event_logger(
        output, console=lambda message: print(message, flush=True)
    )
    render_run_page(output, request, status)
    print(
        f"\nOpen this page during or after the run:\n{output / 'index.html'}\n",
        flush=True,
    )

    def emit(event: str, message: str, data: EventData) -> None:
        logger(event, message, data)
        if event in {
            "model_loading",
            "model_ready",
            "sam_prompt_rewritten",
            "sam_prompt_rewrite_failed",
            "split_completed",
            "sam3_started",
            "sam3_filtering",
            "qwen_started",
            "qwen_completed",
            "candidate_verification_started",
            "candidate_verification_completed",
            "candidate_verification_failed",
            "region_gated",
            "region_scored",
            "iteration_completed",
            "evaluation_failed",
            "search_finished",
        }:
            render_run_page(output, request, status)

    try:
        emit(
            "run_created",
            "Manual run created; all model inputs and decisions will be saved.",
            {
                "query": query,
                "detector_prompt": concept,
                "config": config,
                "image_size": image.size,
            },
        )
        result = run_visual_search(
            np.asarray(image),
            query,
            config=config,
            artifact_dir=output / "stages",
            progress=lambda message: None,
            on_event=emit,
        )
        save_search_result(result, np.asarray(image), output)
        request["detector_prompt"] = result["config"].get("sam_prompt", concept)
        request["config"] = result["config"]
        (output / "request.json").write_text(json.dumps(request, indent=2))
        if args.question_id and not args.no_ground_truth:
            emit(
                "inspection_started",
                "Create ground-truth overlays after inference; annotations were not model inputs.",
                {},
            )
            make_inspection(
                output,
                output / "inspection",
                data_dir=args.data_dir,
                image_id=str(args.image_id),
                question_id=args.question_id,
            )
        status["state"] = "failed" if result["error"] else "completed"
        status["result_path"] = "results.json"
        if result["error"]:
            status["error"] = result["error"]
        emit(
            "run_finished",
            f"Manual run {status['state']}; {len(result.get('warnings', []))} recoverable region warnings.",
            {
                "stop_reason": result["stop_reason"],
                "usage": result["usage"],
                "error": result["error"],
            },
        )
    except KeyboardInterrupt:
        status["state"] = "cancelled"
        status["error"] = (
            "Cancelled by user; saved stages, tree snapshots and logs are preserved."
        )
        emit("run_cancelled", status["error"], {"error": status["error"]})
        render_run_page(output, request, status)
        return 130
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc())
        emit("run_failed", status["error"], {"error": status["error"]})
        render_run_page(output, request, status)
        print(
            f"Review {output / 'index.html'} and {output / 'error.txt'}. For an occupied GPU, retry in a fresh folder with --device cpu.",
            flush=True,
        )
        return 2
    render_run_page(output, request, status)
    print(
        f"\nViewer: {output / 'index.html'}\nReadable log: {output / 'run.log'}\nStructured decisions: {output / 'events.jsonl'}",
        flush=True,
    )
    return 0 if status["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
