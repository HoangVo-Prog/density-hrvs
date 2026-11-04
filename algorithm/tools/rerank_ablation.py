"""Compare candidates from frozen inference pools; evaluate labels only after saving predictions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from ..core.calls import charge_generation
from ..core.data import load_dataset_image
from ..models.qwen_scoring import QWEN_MODEL, QWEN_CHECKPOINTS, load_qwen_scorer
from ..search.reranking import rerank_regions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--question-ids',nargs='*')
    p.add_argument('--model',default=QWEN_MODEL)
    p.add_argument('--model-revision')
    p.add_argument('--cache-dir')
    p.add_argument('--quantization',choices=['none','nf4'],default='none')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    args.model_revision=args.model_revision or QWEN_CHECKPOINTS.get(args.model)
    if not args.model_revision:raise ValueError("An explicit model revision is required")
    selection=json.loads((args.batch/'selection.json').read_text())
    if args.question_ids:selection=[r for r in selection if r['question_id'] in set(args.question_ids)]
    files=[Path(__file__),Path(__file__).parents[1]/'search/reranking.py',
        Path(__file__).parents[1]/'models/comparison.py',Path(__file__).parents[1]/'models/qwen_scoring.py']
    identity={'selection':selection,'batch':str(args.batch.resolve()),'model':args.model,'revision':args.model_revision,'quantization':args.quantization,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    manifest=args.output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Experiment identity changed')
    manifest.write_text(json.dumps(identity,indent=2)+'\n')
    scorer=load_qwen_scorer('cuda',model_id=args.model,revision=args.model_revision,cache_dir=args.cache_dir,quantization=args.quantization)
    for index,item in enumerate(selection,1):
        qid=item['question_id'];folder=args.output/qid;file=folder/'prediction.json'
        if file.exists():continue
        folder.mkdir(parents=True,exist_ok=True)
        source=load_dataset_image(args.data_dir,item['image_id'])
        reportpath=args.batch/qid/'run/results.json';report=json.loads(reportpath.read_text())
        proposals=copy.deepcopy(report['proposals'])
        usage={};settings={**report['config'],'comparative_reranking':True,'rerank_candidate_limit':5}
        state=SimpleNamespace(settings=settings,source_image=source,artifacts=folder,query=report['query'],
            local_scorer=scorer,remaining_final_generations=lambda:2,
            charge_final=lambda stage:charge_generation(usage,stage,2),record_cost=lambda *a:None)
        result={'ranked_regions':proposals};rerank_regions(state,result)
        row={**item,'source_result_sha256':hashlib.sha256(reportpath.read_bytes()).hexdigest(),
            'usage':usage,'reranking':result['reranking'],
            'selected_bbox':result['ranked_regions'][0]['bbox'] if result['reranking']['status']=='selected' else None}
        tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(file)
        print(f"{index}/{len(selection)} {qid}: {row['reranking']['status']}",flush=True)


if __name__=='__main__':main()
