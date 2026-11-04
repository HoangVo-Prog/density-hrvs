"""Draw reference proposals separately from requested targets."""
from pathlib import Path
import html
import json
from PIL import Image, ImageDraw, ImageFont
from ..core.imaging import to_pil_rgb
from ..core.geometry import box_iou


def save_reference_views(result, pixels, output):
    if result['config'].get('detector_prompt_mode') != 'references':
        return
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plan = result['config']['reference_plan']
    all_boxes = result['config']['reference_detections']
    source = to_pil_rgb(pixels)
    palette = ['#ff00aa', '#00ff44', '#ffcc00', '#00ccff', '#ff6600', '#bb88ff']
    combined = source.copy(); combined_draw = ImageDraw.Draw(combined)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', max(15, source.width//110))
    refs = []
    for ri, ref in enumerate(plan['references']):
        unique = []
        for box in sorted((b for b in all_boxes if b['reference_id']==ref['id']), key=lambda b:b['sam3_score'], reverse=True):
            if all(box_iou(box['bbox'], old['bbox']) < .5 for old in unique):
                unique.append(box)
        canvas = source.copy(); draw = ImageDraw.Draw(canvas); color = palette[ri%len(palette)]
        for bi, box in enumerate(unique, 1):
            a,b,c,d = box['bbox']; label = f'R{ri+1}.{bi} {box["sam3_score"]:.3f}'
            for painter in (draw, combined_draw):
                painter.rectangle((a,b,c-1,d-1), outline=color, width=max(4, source.width//600))
                painter.text((a,max(0,b-font.size-3)), label, font=font, fill=color, stroke_width=2, stroke_fill='black')
        canvas.thumbnail((1920,1920)); filename = ref['id']+'.jpg'; canvas.save(output/filename)
        refs.append({**ref, 'boxes': unique, 'image':filename, 'color':color})
    combined.thumbnail((1920,1920));combined.save(output/'all_references.jpg')
    payload = {'question':result.get('final_answer',{}).get('question',result['query']), 'target':plan['target'],
               'references':refs, 'final_answer':result.get('final_answer',{}).get('answer',result.get('abstention',{}).get('answer','Not generated')),
               'note':'Boxes localize reference objects, not the requested target. Final answer uses the existing padded crop and original question.'}
    (output/'references.json').write_text(json.dumps(payload,indent=2)+'\n')
    body = ''.join('<h2>'+html.escape(r['prompt'])+'</h2><p>'+html.escape(r['context'])+f' | {len(r["boxes"])} boxes</p><img src="'+r['image']+'">' for r in refs)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;background:#15171b;color:#eee;margin:30px}img{max-width:100%}</style><h1>Reference detections</h1><p>'+html.escape(payload['question'])+'</p><p>Answer: '+html.escape(payload['final_answer'])+'</p><p>'+html.escape(payload['note'])+'</p>'+body)
