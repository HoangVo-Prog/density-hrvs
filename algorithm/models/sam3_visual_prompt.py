"""Draw identical SAM3 candidate IDs in overview and detail; never use GT boxes."""
from collections import defaultdict
import math
from PIL import Image, ImageDraw, ImageFont
from .spatial_context import build_region_context


def _draw_box(image, box, label, role):
    draw = ImageDraw.Draw(image)
    color = (255, 80, 210) if role == "target" else (255, 165, 40)
    width = 2 if role == "target" else 3
    a, b, c, d = box
    font = ImageFont.load_default(size=12)
    label_y = max(0, b-15) if role == "target" else min(image.height-15, d)
    text_box = draw.textbbox((a, label_y), label, font=font)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.text((a, label_y), label, fill=color, font=font)
    draw.rectangle((a, b, c-1, d-1), outline=color, width=width)


def build_sam3_region_context(source, bbox, mask=None, *, detections=(), node_id=None, max_side=768):
    context = build_region_context(source, bbox, mask, node_id=node_id, max_side=max_side)
    detail = source.convert("RGB").crop(bbox)
    x0, y0, x1, y1 = bbox
    sx, sy = context["overview"].width/source.width, context["overview"].height/source.height
    groups = {}; counts = defaultdict(int); legend = []; boxes = []
    for detection in detections:
        role = detection.get("proposal_role", "reference")
        if role not in {"target", "reference"}: role = "reference"
        prompt = detection.get("prompt", "object")
        key = (role, detection.get("reference_id", prompt))
        if key not in groups:
            prefix = ("T" if role == "target" else "R") + str(1 + sum(k[0] == role for k in groups))
            groups[key] = prefix; legend.append({"id": prefix, "role": role, "prompt": prompt})
        values = detection["bbox"]
        if len(values) != 4 or not all(math.isfinite(float(v)) for v in values): continue
        a,b,c,d = values
        a,b,c,d = max(0, math.floor(a)), max(0, math.floor(b)), min(detail.width, math.ceil(c)), min(detail.height, math.ceil(d))
        if a >= c or b >= d: continue
        counts[key] += 1; label = groups[key] + "." + str(counts[key])
        global_box = [x0+a, y0+b, x0+c, y0+d]
        oa,ob,oc,od = [round(v*s) for v,s in zip(global_box, [sx,sy,sx,sy])]
        oa,ob = min(context["overview"].width-1,oa), min(context["overview"].height-1,ob)
        oc,od = min(context["overview"].width,max(oa+1,oc)), min(context["overview"].height,max(ob+1,od))
        boxes.append({"id":label,"role":role,"prompt":prompt,"local_bbox_xyxy":[a,b,c,d],
            "global_bbox_xyxy":global_box,"overview_bbox_xyxy":[oa,ob,oc,od],
            "sam3_score":detection.get("sam3_score"),"sam3_selection":detection.get("sam3_selection")})
    for box in sorted(boxes, key=lambda box: box["role"] == "target"):
        _draw_box(detail, box["local_bbox_xyxy"], box["id"], box["role"])
        _draw_box(context["overview"], box["overview_bbox_xyxy"], box["id"], box["role"])
    context["detail"] = detail
    context["metadata"].update(sam3_box_legend=[entry for entry in legend if any(box["id"].startswith(entry["id"] + ".") for box in boxes)], sam3_boxes=boxes,
        direction_convention="displayed-image axes, not object-centric orientation",
        box_colors={"target":"magenta", "reference":"orange", "region":"cyan"})
    return context
