"""Command-line entry point; imports model-heavy modules only inside main()."""

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    """Run the complete algorithm from an image and query, saving a reviewable trace."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DINO/SLIC + SAM3 candidates + local Qwen3-VL-2B verification"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input RGB image path")
    source.add_argument(
        "--image-id", help="Image ID in data_new, e.g. 0063; ZIPs are read directly"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "data_new",
    )
    parser.add_argument(
        "--query", required=True, help="Target description with required attributes"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/visual_search")
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument(
        "--strategy", choices=("agent-uct", "uct", "greedy"), default="agent-uct"
    )
    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=32,
        help="Physical HTTP attempts including retries; 0 disables this limit",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Root depth is zero; use one for a single-pass search",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--branching-factor", type=int, default=5)
    parser.add_argument(
        "--max-proposals-per-node",
        type=int,
        default=16,
        help="Maximum SAM3 boxes retained per region before NMS; 0 keeps all",
    )
    parser.add_argument(
        "--candidate-check-limit",
        type=int,
        default=2,
        help="Maximum SAM3 candidates verified per region; 0 disables verification",
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
        "--sam-prompt",
        "--detector-prompt",
        dest="sam_prompt",
        default="",
        help="Object concept for SAM3",
    )
    parser.add_argument(
        "--max-local-scores",
        type=int,
        default=32,
        help="Local Qwen generation budget including repairs",
    )
    parser.add_argument("--cost-weight", type=float, default=0.2)
    parser.add_argument(
        "--context-padding",
        type=int,
        default=384,
        help="Minimum original-image context margin around a candidate",
    )
    parser.add_argument(
        "--detail-padding",
        type=int,
        default=16,
        help="Minimum source-pixel margin around the enlarged detail",
    )
    parser.add_argument(
        "--verification-max-side",
        type=int,
        default=1536,
        help="Context/detail input width; at least 512 pixels",
    )
    parser.add_argument(
        "--stop-on-match",
        action="store_true",
        help="Stop at a localized Yes with rating >= 0.9",
    )
    args = parser.parse_args()
    from ..core.data import load_dataset_image
    from ..search.runner import run_visual_search, save_search_result
    from ..search.config import default_search_config

    config = default_search_config()
    config.update(
        device=args.device,
        strategy=args.strategy,
        max_api_calls=args.max_api_calls,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        branching_factor=args.branching_factor,
        max_proposals_per_node=args.max_proposals_per_node,
        sam_prompt=args.sam_prompt,
        cost_weight=args.cost_weight,
        stop_on_match=args.stop_on_match,
        context_padding=args.context_padding,
        detail_padding=args.detail_padding,
        verification_max_side=args.verification_max_side,
        max_local_scores=args.max_local_scores,
        sam3_first=not args.legacy_region_qwen,
        candidate_check_limit=args.candidate_check_limit,
        candidate_query_mode=args.candidate_query_mode,
    )
    try:
        if args.image_id:
            image = np.asarray(load_dataset_image(args.data_dir, args.image_id))
        else:
            with Image.open(args.image) as source_image:
                image = np.asarray(source_image.convert("RGB"))
        result = run_visual_search(
            image, args.query, config=config, artifact_dir=args.output_dir / "stages"
        )
        path = save_search_result(result, image, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Search failed: {exc}\n")
    print(f"Saved {path}")
    print(
        f"Stop: {result['stop_reason']} | HTTP calls: {result['usage']['api_calls']} | evaluated proposals: {len(result['ranked_regions'])}"
    )
    if result["error"]:
        print(result["error"])
        return 2
    return 0
