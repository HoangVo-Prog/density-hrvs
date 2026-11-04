"""Project inference-only fields from legacy samples or the official image export."""
import json
from pathlib import Path


def _records(root):
    root=Path(root)
    legacy=sorted(root.glob('hrvs_bbox_samples_*.json'))
    if legacy:
        for row in json.loads(legacy[-1].read_text()):
            yield str(row['image_id']), row
        return
    payload=json.loads((root/'annotations.json').read_text())
    for image in payload['images']:
        image_id=Path(image.get('file_name') or str(image['image_id'])).stem
        for qa in image.get('qa_annotations',[]):
            yield image_id, qa


def load_inference_questions(root):
    # Explicitly exclude answers, points, category/color attributes and target boxes.
    rows=[];seen=set()
    for image_id, item in _records(root):
        question_id=item['question_id']
        if not isinstance(question_id,str) or not question_id or Path(question_id).name != question_id:
            raise ValueError('Question ID must be a safe nonempty filename')
        if question_id in seen:raise ValueError(f'Duplicate question ID: {question_id}')
        seen.add(question_id)
        question=item['question']
        if not isinstance(question,str) or not question.strip():raise ValueError('Question must be text')
        rows.append({'image_id':image_id,'question_id':question_id,'question':question})
    return rows


def load_evaluation_rows(root):
    # Call only after predictions have been saved. It is not used by inference.
    rows=[]
    for image_id,item in _records(root):
        box=item.get('target_bbox')
        parsed_box = None
        if isinstance(box, dict) and all(k in box for k in ('x', 'y', 'width', 'height')):
            parsed_box = {k: box[k] for k in ('x', 'y', 'width', 'height')}
        rows.append({'image_id':image_id,'question_id':item['question_id'],
            'question':item['question'],'answer':item.get('answer'),
            'target_bbox':parsed_box})
    return rows
