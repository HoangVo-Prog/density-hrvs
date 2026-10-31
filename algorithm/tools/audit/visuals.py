"""Post-run visual review cards for every audited question, with annotated truth."""

import argparse
from functools import lru_cache
import html
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ...core.data import load_dataset_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("new_outputs/sam3_region_qwen_all")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_new"))
    args = parser.parse_args()
    rows = json.loads((args.run_root / "audit/audit.json").read_text())["questions"]
    output = args.run_root / "audit/visuals"
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

    @lru_cache(maxsize=1)
    def source(image_id: str) -> Image.Image:
        return load_dataset_image(args.data_dir, image_id)

    links = []
    for row in rows:
        qid = row["question_id"]
        original = source(row["image_id"])
        gt = row["target_bbox"]
        folder = args.run_root / qid
        result_file = folder / "run/results.json"
        report = json.loads(result_file.read_text()) if result_file.exists() else {}
        predictions = report.get("proposals", [])
        selected = predictions[0] if predictions else None
        node = next(
            (n for n in row["nodes"] if n["node_id"] == row["winner_node"]), None
        )
        scene = original.copy()
        drawing = ImageDraw.Draw(scene)
        line = max(3, round(max(original.size) / 450 * 2))
        drawing.rectangle(gt, outline="lime", width=line)
        if selected:
            drawing.rectangle(selected["bbox"], outline="red", width=line)
        region = original.crop(node["bbox"]) if node else original.copy()
        if node:
            x, y, _, _ = node["bbox"]
            draw = ImageDraw.Draw(region)
            if selected:
                a, b, c, d = selected["bbox"]
                draw.rectangle(
                    (a - x, b - y, c - x, d - y),
                    outline="red",
                    width=max(2, round(max(region.size) / 450 * 2)),
                )
            a, b, c, d = gt
            if max(a, x) < min(c, node["bbox"][2]) and max(b, y) < min(
                d, node["bbox"][3]
            ):
                draw.rectangle(
                    (a - x, b - y, c - x, d - y),
                    outline="lime",
                    width=max(2, round(max(region.size) / 450 * 2)),
                )
        margin = max(512, round(3 * max(gt[2] - gt[0], gt[3] - gt[1])))
        window = (
            max(0, gt[0] - margin),
            max(0, gt[1] - margin),
            min(original.width, gt[2] + margin),
            min(original.height, gt[3] + margin),
        )
        context = original.crop(window)
        ImageDraw.Draw(context).rectangle(
            (
                gt[0] - window[0],
                gt[1] - window[1],
                gt[2] - window[0],
                gt[3] - window[1],
            ),
            outline="lime",
            width=max(2, round(max(context.size) / 450 * 2)),
        )
        detail = original.crop(
            (
                max(0, gt[0] - 8),
                max(0, gt[1] - 8),
                min(original.width, gt[2] + 8),
                min(original.height, gt[3] + 8),
            )
        )
        canvas = Image.new("RGB", (1800, 1050), "#101827")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (15, 10),
            f"{qid} | {row['diagnosis']} | top1 IoU={row['top1_iou']:.3f} best={row['best_iou']:.3f}",
            font=font,
            fill="white",
        )
        for i, line_text in enumerate(textwrap.wrap(row["question"], 145)[:2]):
            draw.text((15, 43 + i * 27), line_text, font=font, fill="white")
        draw.text(
            (15, 105),
            f"GT category={row['category']}; annotated answer={row['annotated_answer']}; detector concept={row['concept']}",
            font=small,
            fill="#7dd3fc",
        )
        titles = (
            "Full image: green=GT, red=top box",
            "Winning Qwen region (marks added for audit)",
            "GT context (audit only)",
            "Annotated target detail (audit only)",
        )
        for index, (picture, title) in enumerate(
            zip((scene, region, context, detail), titles)
        ):
            left = 450 * index
            draw.text((left + 8, 145), title, font=small, fill="white")
            rendered = ImageOps.contain(picture, (438, 600))
            canvas.paste(
                rendered,
                (
                    left + 6 + (438 - rendered.width) // 2,
                    180 + (600 - rendered.height) // 2,
                ),
            )
        if node:
            draw.text(
                (15, 800),
                f"Winning node {node['node_id']}, depth {node['depth']}, Qwen={node['qwen_score']:.2f}, region reward={node['combined_score']:.3f}; target coverage={node['target_coverage']:.2f}",
                font=small,
                fill="white",
            )
            for i, line_text in enumerate(
                textwrap.wrap("Qwen reason: " + node["qwen_reason"], 170)[:5]
            ):
                draw.text((15, 835 + i * 25), line_text, font=small, fill="white")
        else:
            draw.text(
                (15, 800),
                "No exported winning region. " + row["error"],
                font=small,
                fill="white",
            )
        draw.text(
            (15, 1010),
            "Annotations added only for post-run review. These marked cards were not model inputs.",
            font=small,
            fill="#7dd3fc",
        )
        canvas.save(output / f"{qid}.jpg", quality=88)
        links.append(
            f'<figure><a href="{qid}.jpg"><img loading="lazy" src="{qid}.jpg" style="width:100%"></a><figcaption>{html.escape(qid)} <a href="../../{qid}/stages/tree/index.html">Tree</a></figcaption></figure>'
        )
    (output / "index.html").write_text(
        '<html><meta charset="utf-8"><style>body{max-width:1800px;margin:auto;font:18px system-ui}figure{margin:25px 0}</style><h1>All-question visual audit</h1>'
        + "".join(links)
    )
    print(f"Saved {len(rows)} review cards to {output}")


if __name__ == "__main__":
    main()
