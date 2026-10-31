"""Reconstruct saved runs as chronological visual walkthroughs; no inference."""

import argparse
import html
import json
import os
from pathlib import Path
from typing import TypedDict

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..core.data import load_dataset_image


class Asset(TypedDict):
    """A saved or post-run visual with an explicit role label."""

    label: str  # Distinguishes actual inputs, diagnostics and ground-truth overlays.
    path: str  # Path relative to the walkthrough HTML file.


class Step(TypedDict):
    """Chronological event reconstructed from the immutable saved trace."""

    title: str
    explanation: str
    assets: list[Asset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_new"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report = json.loads((run / "run/results.json").read_text())
    inspection = json.loads((run / "inspection/inspection.json").read_text())
    output = run / "walkthrough"
    output.mkdir(exist_ok=True)
    original = load_dataset_image(args.data_dir, inspection["image_id"])
    gt = inspection["target_bbox"]
    root = run / "stages/nodes/node_0000"
    metrics = json.loads((root / "slic_metrics.json").read_text())
    split = json.loads((root / "split.json").read_text())
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(30, original.width // 70)
    )
    width = max(5, original.width // 400)

    def asset(label: str, path: Path) -> Asset:
        return {"label": label, "path": os.path.relpath(path, output)}

    def mark(image: Image.Image, box: list, caption: str, color: str) -> None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=color, width=width)
        x, y = box[0], max(0, box[1] - font.size - 10)
        area = draw.textbbox((x, y), caption, font=font)
        draw.rectangle(area, fill="black")
        draw.text((x, y), caption, font=font, fill=color)

    def display(image: Image.Image, name: str) -> Path:
        image.thumbnail((1800, 1400))
        path = output / name
        image.save(path, quality=92)
        return path

    annotated = original.copy()
    mark(annotated, gt, "GROUND TRUTH — audit only", "#00ff55")
    source_path = display(annotated, "00_original_with_ground_truth.jpg")
    margin = max(350, round(4 * max(gt[2] - gt[0], gt[3] - gt[1])))
    box = (
        max(0, gt[0] - margin),
        max(0, gt[1] - margin),
        min(original.width, gt[2] + margin),
        min(original.height, gt[3] + margin),
    )
    context = original.crop(box)
    ImageDraw.Draw(context).rectangle(
        (gt[0] - box[0], gt[1] - box[1], gt[2] - box[0], gt[3] - box[1]),
        outline="#00ff55",
        width=4,
    )
    context_path = display(
        ImageOps.contain(context, (1000, 1000)), "00_ground_truth_context.jpg"
    )
    overview = original.copy()
    for node in report["nodes"][1:]:
        mark(overview, node["bbox"], f"N{node['node_id']}", "#ffaf40")
    mark(overview, gt, "GROUND TRUTH", "#00ff55")
    candidate_path = display(overview, "03_all_candidate_region_rectangles.jpg")
    steps: list[Step] = [
        {
            "title": "0. Input image and task",
            "explanation": f"Original image: {original.width} × {original.height}. Ground truth is {gt}; annotated answer: {inspection['annotated_answer']}. Green markings here are added for inspection only and were not provided to the algorithm.",
            "assets": [
                asset(
                    "Original image with ground-truth marker — audit overlay",
                    source_path,
                ),
                asset(
                    "Ground-truth context and wristband — audit crop, not model input",
                    context_path,
                ),
            ],
        },
        {
            "title": "1. DINOv3 feature extraction",
            "explanation": "The full-image root was encoded with DINOv3. The three-channel PCA map below was used by SLIC/K-means. Its colors are visual features, not query scores or ground truth.",
            "assets": [
                asset(
                    "Saved DINO PCA map — actual clustering feature visualization",
                    root / "dino_pca.png",
                )
            ],
        },
        {
            "title": "2. SLIC segmentation",
            "explanation": f"Tested requested segment counts: {[m['n_segments'] for m in metrics['history']]}. Minimum mean entropy selected n_segments={metrics['metrics']['n_segments']}, producing {metrics['metrics']['actual_regions']} actual superpixels. Mean entropy={metrics['metrics']['mean_entropy']:.4f}.",
            "assets": [
                asset(
                    "Selected SLIC boundaries over the image",
                    root / "slic_selected_boundaries.jpg",
                ),
                asset(
                    "Selected SLIC region IDs, displayed in arbitrary colors",
                    root / "slic_selected_colors.png",
                ),
            ],
        },
        {
            "title": "3. Group superpixels into candidate search regions",
            "explanation": f"K={len(report['nodes'][0]['children'])} child regions were created. These are appearance/spatial groups, not relevance-ranked top-K regions. Spatial fallback was {split['used_spatial_fallback']}; this run kept its semantic groups. Orange rectangles below are candidate REGION bounds; green is the annotated target, added after the run. Region rectangles can overlap even though their masks partition pixels.",
            "assets": [
                asset("K-means label map", root / "kmeans_colors.png"),
                asset("K-means boundaries", root / "kmeans_boundaries.jpg"),
                asset(
                    "All candidate region rectangles with node IDs — audit overlay",
                    candidate_path,
                ),
            ],
        },
    ]
    for index, path in enumerate(sorted(root.glob("semantic_cluster_*.jpg"))):
        steps[3]["assets"].append(
            asset(
                f"Semantic mask group {index} — clustering diagnostic, not Qwen input",
                path,
            )
        )
    for event in report["trace"]:
        node = report["nodes"][event["node_id"]]
        folder = run / f"stages/nodes/node_{node['node_id']:04d}"
        selected = original.copy()
        mark(selected, node["bbox"], f"ATTEMPTED REGION N{node['node_id']}", "#00d9ff")
        mark(selected, gt, "GROUND TRUTH — audit only", "#00ff55")
        picture = display(
            selected, f"iteration_{event['iteration']:03d}_selected_region.jpg"
        )
        intersects = max(node["bbox"][0], gt[0]) < min(node["bbox"][2], gt[2]) and max(
            node["bbox"][1], gt[1]
        ) < min(node["bbox"][3], gt[3])
        steps.append(
            {
                "title": f"4. Iteration {event['iteration']}: select region N{node['node_id']}",
                "explanation": f"Path: {event['path']}; depth {node['depth']}; region bbox {node['bbox']}. Cyan is the entire selected REGION; green is the ground-truth OBJECT box. Target intersects this region: {intersects}. Selection terms: Q={event['quality']:.4f}, exploration={event['exploration_bonus']:.4f}, estimated cost={event['estimated_remaining_seconds']:.2f}s, index={event['selection_index']:.4f}. Initially all children are unvisited; this is an exploration choice, not a positive Qwen relevance judgment.",
                "assets": [
                    asset(
                        "Full scene: cyan attempted region versus green ground truth",
                        picture,
                    ),
                    asset(
                        "Exact unmasked region supplied to Qwen — original saved pixels",
                        folder / "region_input.png",
                    ),
                ],
            }
        )
        score = folder / "region_score.json"
        detections = folder / "detections.json"
        if score.exists():
            assessment = json.loads(score.read_text())
            note = f"Valid Qwen rating: {assessment['score']:.3f}, {assessment['answer']}. Reason: {assessment['reason']}"
        else:
            note = f"No valid region assessment was saved. This historical run stopped with: {report['error']}. Local generations used: {report['usage'].get('local_score_calls', 0)}. Raw invalid replies were not retained by this old version, so their exact text cannot be reconstructed."
        steps.append(
            {
                "title": "5. Qwen whole-region assessment",
                "explanation": note,
                "assets": [],
            }
        )
        if detections.exists():
            boxes = json.loads(detections.read_text())
            overlay = folder / "sam3_distinct_boxes.jpg"
            steps.append(
                {
                    "title": "6. SAM3 bounding boxes",
                    "explanation": f"{len(boxes)} above-threshold detections were saved before NMS. Coordinates in detections.json are relative to the selected crop. This step follows Qwen assessment in the saved implementation.",
                    "assets": [asset("Saved SAM3 box overlay", overlay)]
                    if overlay.exists()
                    else [],
                }
            )
        else:
            steps.append(
                {
                    "title": "6. SAM3 bounding boxes — NOT REACHED",
                    "explanation": "SAM3 did not run for this region. In this version Qwen assessment happens before detection, and the invalid Qwen reply stopped execution. No SAM3 boxes exist for this saved run; displaying generated boxes here would require a new, separately labeled rerun.",
                    "assets": [],
                }
            )
    steps.append(
        {
            "title": "7. Final state",
            "explanation": f"Stop reason: {report['stop_reason']}. Completed region evaluations: {sum(t['completed'] for t in report['trace'])}. Saved object proposals: {len(report['proposals'])}. No valid reward was backed up for an incomplete evaluation. Other child regions remained unvisited. The failed first region is not a winning detection.",
            "assets": [],
        }
    )
    records = {
        "question": inspection["question"],
        "source": "Saved historical run only; no new inference was performed",
        "ground_truth_bbox": gt,
        "steps": steps,
        "trace": report["trace"],
    }
    (output / "steps.json").write_text(json.dumps(records, indent=2))
    sections = []
    for index, step in enumerate(steps):
        figures = "".join(
            f'<figure><figcaption>{html.escape(a["label"])}</figcaption><a href="{a["path"]}"><img loading="lazy" src="{a["path"]}"></a></figure>'
            for a in step["assets"]
        )
        sections.append(
            f'<section id="step-{index}"><h2>{html.escape(step["title"])}</h2><p>{html.escape(step["explanation"])}</p>{figures}</section>'
        )
    history = "".join(
        f"<tr><td>{r['n_segments']}</td><td>{r['actual_regions']}</td><td>{r['mean_entropy']:.4f}</td></tr>"
        for r in metrics["history"]
    )
    children = ""
    for node in report["nodes"][1:]:
        b = node["bbox"]
        contains = b[0] <= gt[0] and b[1] <= gt[1] and b[2] >= gt[2] and b[3] >= gt[3]
        attempted = any(t["node_id"] == node["node_id"] for t in report["trace"])
        status = (
            "attempted; assessment failed"
            if attempted and not node["evaluated"]
            else "evaluated"
            if node["evaluated"]
            else "not visited"
        )
        children += f"<tr><td>N{node['node_id']}</td><td>{b}</td><td>{contains}</td><td>{status}</td></tr>"
    nav = " ".join(
        f'<a href="#step-{i}">{html.escape(s["title"])}</a>'
        for i, s in enumerate(steps)
    )
    page = f'<html><meta charset="utf-8"><style>body{{font:19px system-ui;line-height:1.55;max-width:1600px;margin:25px auto;padding:20px;background:#101827;color:#eee}}a{{color:#7dd3fc}}img{{max-width:100%;max-height:900px}}figure{{border:1px solid #64748b;padding:12px;margin:20px 0}}section{{border-top:2px solid #64748b;margin-top:35px}}td,th{{padding:9px;border:1px solid #64748b}}nav a{{display:inline-block;margin:5px 14px 5px 0}}.alert{{border:2px solid #ff7f50;padding:15px}}</style><h1>{html.escape(run.name)}: actual run, step by step</h1><p>{html.escape(inspection["question"])}</p><p class="alert">Historical run failed before SAM3. Cyan = attempted region. Green = ground truth, added only for inspection. No final predicted box exists.</p><nav>{nav}</nav>{"".join(sections)}<h2>All initial candidate regions</h2><table><tr><th>Node</th><th>Global xyxy region bbox</th><th>Contains GT rectangle?</th><th>Saved status</th></tr>{children}</table><h2>SLIC selection history</h2><table><tr><th>Requested count</th><th>Actual superpixels</th><th>Mean entropy</th></tr>{history}</table><p><a href="../stages/tree/index.html">Original saved tree</a> | <a href="../stages/nodes/node_0000/index.html">Original root stages</a> | <a href="steps.json">Chronological data</a></p></html>'
    (output / "index.html").write_text(page)
    print(output / "index.html")
    print(
        f"Historical evaluations: {len(report['trace'])}; saved detections: {len(report['proposals'])}; no new model calls"
    )


if __name__ == "__main__":
    main()
