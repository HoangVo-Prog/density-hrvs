"""Paired GPU answering experiment with frozen, label-free selected boxes.

Predictions are saved before an independent exact-answer diagnostic is computed.
This tool does not run search or choose boxes using ground-truth overlap.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

from PIL import Image
import torch

from ..core.calls import charge_generation
from ..core.data import load_dataset_image
from ..models.qwen_scoring import QWEN_MODEL, QWEN_REVISION, load_qwen_scorer
from ..search.answering import answer_top_region


def normalized(text: str) -> str:
    value = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'^(?:a|an|the)\s+', '', ' '.join(value.split()))


def summarize(output: Path, batch: Path) -> dict:
    labels = {r['question_id']:r for r in json.loads((batch/'graded_semantic.json').read_text())['rows']}
    rows = []
    for path in sorted(output.glob('*/prediction.json')):
        row = json.loads(path.read_text())
        gt = labels[row['question_id']]['ground_truth']
        rows.append({**row, 'ground_truth':gt,
            'exact_diagnostic': {view: normalized(value['answer']) == normalized(gt)
                                  for view,value in row['views'].items()}})
    views = sorted({v for r in rows for v in r['views']})
    report = {'questions':len(rows),
        'metric':'Exact normalized complete answer; conservative diagnostic, not the saved semantic rubric.',
        'exact_correct':{v:sum(r['exact_diagnostic'].get(v,False) for r in rows) for v in views},
        'rows':rows}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='rows'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--question-ids',nargs='*')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--max-side',type=int,default=1536)
    parser.add_argument('--views',nargs='+',choices=['legacy_crop','marked_detail'],
                        default=['legacy_crop','marked_detail'])
    args=parser.parse_args()
    selection=json.loads((args.selection or args.batch/'selection.json').read_text())
    if args.question_ids:
        selection=[r for r in selection if r['question_id'] in set(args.question_ids)]
    if args.limit:selection=selection[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    identity={'batch':str(args.batch.resolve()), 'selection':selection,'views':args.views,
        'model':QWEN_MODEL,'revision':QWEN_REVISION,'device':args.device,'max_side':args.max_side,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(__file__).parents[1]/'search/answering.py',
                      Path(__file__).parents[1]/'models/evidence.py',
                      Path(__file__).parents[1]/'models/qwen_scoring.py']},
        'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','pillow']}}
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise ValueError('Experiment identity changed; choose a new output directory')
    manifest.write_text(json.dumps(identity,indent=2)+'\n')
    tick=time.perf_counter();scorer=load_qwen_scorer(args.device)
    print(f'Model ready in {time.perf_counter()-tick:.1f}s',flush=True)
    for index,item in enumerate(selection,1):
        qid=item['question_id'];folder=args.output/qid;checkpoint=folder/'prediction.json'
        if checkpoint.exists():
            print(f'{index}/{len(selection)} {qid}: checkpoint exists',flush=True)
            continue
        report_path=args.batch/qid/'run/results.json'
        report=json.loads(report_path.read_text())
        folder.mkdir(parents=True,exist_ok=True)
        source=load_dataset_image(args.data_dir,item['image_id'])
        # The box comes only from the existing inference result, never inspection/GT.
        top=report['proposals'][:1]
        row={'question_id':qid,'question':item['question'],
             'frozen_bbox':top[0]['bbox'] if top else None,
             'source_result_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'views':{}}
        for view in args.views:
            usage={};settings={**report['config'],'answer_view':view,
                'verification_max_side':args.max_side,
                # Isolate presentation: eligibility is held fixed and top-1 never changes.
                'require_verified_answer':False}
            state=SimpleNamespace(settings=settings,source_image=source,question=item['question'],
                usage=usage,local_scorer=scorer,
                charge_answer=lambda:charge_generation(usage,'answer'))
            torch.cuda.reset_peak_memory_stats() if args.device.startswith('cuda') else None
            start=time.perf_counter();answer=answer_top_region(state,{'ranked_regions':top})
            elapsed=time.perf_counter()-start
            if answer is not None:
                Image.fromarray(answer.pop('input_pixels')).save(folder/f'{view}.png')
                answer['image_path']=f'{view}.png'
            row['views'][view]={'answer':answer['answer'] if answer else 'Uncertain',
                'elapsed_seconds':elapsed,'usage':usage,
                'peak_gpu_bytes':torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                'detail':answer}
            print(f'{index}/{len(selection)} {qid} {view}: {row["views"][view]["answer"]} ({elapsed:.1f}s)',flush=True)
        temporary=checkpoint.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(checkpoint)
        print(json.dumps(summarize(args.output,args.batch)),flush=True)


if __name__=='__main__':
    main()
