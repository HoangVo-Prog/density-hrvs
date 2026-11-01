"""Inspection and result exports, separate from search decisions."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ..core.contracts import (
    CandidateCheck,
    ClusterReport,
    NodeReport,
    ProposalReport,
    RegionScoreComponents,
    SAM3Detection,
    SearchNode,
    SearchReport,
    SearchResult,
)
from ..models.evidence import evidence_info
from ..core.diagnostics import (
    save_image,
)
from ..core.imaging import to_pil_rgb


def save_search_result(
    result: SearchResult, image: NDArray, output_dir: str | Path
) -> Path:
    """Save trace, masks, raw crops, actual scoring sheets, and an image preview.

    Args:
        result: Complete or budget-limited result returned by run_visual_search.
        image: The same original RGB image used for the search.
        output_dir: Destination for results.json, masks.npz, crop folders and preview.

    Returns:
        Absolute path to the saved JSON report. Credentials are never serialized.
    """
    output = Path(output_dir).resolve()
    (output / "clusters").mkdir(parents=True, exist_ok=True)
    (output / "proposals").mkdir(exist_ok=True)
    (output / "evidence").mkdir(exist_ok=True)
    (output / "region_inputs").mkdir(exist_ok=True)
    nodes: list[NodeReport] = []
    masks: dict[str, NDArray[np.bool_]] = {}
    for node in result["nodes"]:
        key = f"node_{node['node_id']}"
        masks[key] = node["mask"]
        nodes.append(
            {
                "node_id": node["node_id"],
                "parent_id": node["parent_id"],
                "depth": node["depth"],
                "bbox": node["bbox"],
                "active_pixels": int(node["mask"].sum()),
                "children": node["children"],
                "visits": node["visits"],
                "mean_quality": node["reward_sum"] / node["visits"]
                if node["visits"]
                else 0.0,
                "evaluated": node["evaluated"],
                "exhausted": node["exhausted"],
                "mask_key": key,
            }
        )
        if "evaluation_error" in node:
            nodes[-1]["evaluation_error"] = node["evaluation_error"]
    clusters: list[ClusterReport] = []
    for cluster in result["ranked_clusters"]:
        relative = f"clusters/node_{cluster['cluster_id']}.png"
        to_pil_rgb(cluster["image"]).save(output / relative)
        input_path = f"region_inputs/node_{cluster['cluster_id']}.png"
        to_pil_rgb(cluster["image"]).save(output / input_path)
        cluster_report: ClusterReport = {
            "cluster_id": cluster["cluster_id"],
            "bbox": cluster["bbox"],
            "score": cluster["score"],
            "verifier": cluster["verifier"],
            "image_path": relative,
            "scoring_input_image_path": input_path,
        }
        if "score_components" in cluster:
            cluster_report["score_components"] = cluster["score_components"]
        clusters.append(cluster_report)
    proposals: list[ProposalReport] = []
    for index, proposal in enumerate(result["ranked_regions"]):
        relative = f"proposals/{index:04d}.png"
        key = f"proposal_{index}"
        masks[key] = proposal["mask"]
        to_pil_rgb(proposal["image"]).save(output / relative)
        proposal_report: ProposalReport = {
            "cluster_id": proposal["cluster_id"],
            "bbox": proposal["bbox"],
            "local_bbox": proposal["local_bbox"],
            "score": proposal["score"],
            "verifier": proposal["verifier"],
            "image_path": relative,
            "sam3_score": proposal["sam3_score"],
            "cluster_score": proposal["cluster_score"],
            "mask_key": key,
        }
        if proposal.get("scoring_scope") == "region":
            proposal_report["scoring_scope"] = "region"
            proposal_report["scoring_input_image_path"] = (
                f"region_inputs/node_{proposal['cluster_id']}.png"
            )
        elif proposal.get("scoring_scope") == "proposal":
            proposal_report["scoring_scope"] = "proposal"
        for name in ("candidate_rank", "candidate_score", "candidate_verified"):
            if name in proposal:
                proposal_report[name] = proposal[name]
        if "score_components" in proposal:
            proposal_report["score_components"] = proposal["score_components"]
        if "evidence" in proposal:
            view = proposal["evidence"]
            view_path = f"evidence/{index:04d}.png"
            to_pil_rgb(view["image"]).save(output / view_path)
            proposal_report["evidence_image_path"] = view_path
            proposal_report["evidence_info"] = evidence_info(view)
        proposals.append(proposal_report)
    report: SearchReport = {
        "query": result["query"],
        "image_size": to_pil_rgb(image).size,
        "config": result["config"],
        "usage": result["usage"],
        "stop_reason": result["stop_reason"],
        "error": result["error"],
        "elapsed_seconds": result["elapsed_seconds"],
        "nodes": nodes,
        "trace": result["trace"],
        "clusters": clusters,
        "proposals": proposals,
    }
    if result.get("warnings"):
        report["warnings"] = result.get("warnings", [])
    if "final_answer" in result:
        final = result["final_answer"]
        source = to_pil_rgb(image)
        answer_path = "final_answer_region.png"
        source.crop(final["crop_bbox"]).save(output / answer_path)
        final["image_path"] = answer_path
        report["final_answer"] = final
    np.savez_compressed(output / "masks.npz", allow_pickle=False, **masks)
    preview = to_pil_rgb(image)
    draw = ImageDraw.Draw(preview)
    for rank, proposal in enumerate(result["ranked_regions"][:20], 1):
        verified = proposal.get("candidate_verified")
        if verified is None:
            verified = proposal["verifier"]["answer"] == "Yes"
        color = "lime" if verified else "orange"
        x1, y1, x2, y2 = proposal["bbox"]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{rank}: {proposal['score']:.2f}", fill=color)
    preview.save(output / "preview.jpg", quality=95)
    path = output / "results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path


def save_node_input(
    rgb: NDArray, node: SearchNode, pixels: NDArray, folder: Path | None
) -> None:
    """Save exact model pixels and a separate full-scene region overlay."""
    x0, y0, x_end, y_end = node["bbox"]
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(folder / "region_input.png")
        np.savez_compressed(
            folder / "region_mask.npz",
            mask=node["mask"],
            bbox=np.asarray(node["bbox"]),
        )
        preview = to_pil_rgb(rgb)
        preview.thumbnail((1536, 1024))
        sx, sy = preview.width / rgb.shape[1], preview.height / rgb.shape[0]
        drawing = ImageDraw.Draw(preview)
        drawing.rectangle(
            (x0 * sx, y0 * sy, (x_end - 1) * sx, (y_end - 1) * sy),
            outline="cyan",
            width=4,
        )
        drawing.text(
            (x0 * sx, max(0, y0 * sy - 15)),
            f"Region N{node['node_id']}",
            fill="cyan",
            stroke_width=1,
            stroke_fill="black",
        )
        preview.save(folder / "selected_region_scene.jpg", quality=90)


def save_node_detections(
    pixels: NDArray,
    detections: list[SAM3Detection],
    unique: list[SAM3Detection],
    components: RegionScoreComponents,
    folder: Path | None,
    *,
    checks: list[CandidateCheck] | None = None,
) -> None:
    """Export raw retained boxes, reward terms, checks, and box overlays."""
    if folder is not None:
        (folder / "detections.json").write_text(
            json.dumps(
                [
                    {k: v for k, v in d.items() if k not in {"image", "mask"}}
                    for d in detections
                ],
                indent=2,
            )
        )
        (folder / "region_score_components.json").write_text(
            json.dumps(components, indent=2)
        )
        if checks is not None:
            serializable: list[dict[str, object]] = []
            for check in checks:
                item: dict[str, object] = {
                    key: value for key, value in check.items() if key != "evidence"
                }
                if "evidence" in check:
                    item["evidence"] = evidence_info(check["evidence"])
                serializable.append(item)
            (folder / "candidate_checks.json").write_text(
                json.dumps(serializable, indent=2)
            )
        overlay = Image.fromarray(pixels)
        draw = ImageDraw.Draw(overlay)
        for index, detection in enumerate(unique):
            a, b, c, d = detection["bbox"]
            draw.rectangle(
                (a, b, c - 1, d - 1),
                outline="orange",
                width=max(2, overlay.width // 600),
            )
            draw.text(
                (a, b), f"{index}: {detection['sam3_score'] or 0:.2f}", fill="red"
            )
        save_image(folder / "sam3_distinct_boxes.jpg", np.asarray(overlay))
