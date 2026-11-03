"""Exercise native part refinement on frozen inference-selected entity boxes."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.calls import charge_generation
from ..core.data import load_dataset_image
from ..models.qwen_scoring import load_qwen_scorer
from ..models.sam3 import load_sam3_model, exec_sam3_on_image
from ..search.runtime import RunState
from ..search.parts import refine_answer_part
from ..search.answering import answer_top_region


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--question-ids',nargs='+',required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    questions=[q for q in json.loads((args.batch/'selection.json').read_text())
               if q['question_id'] in set(args.question_ids)]
    qwen=load_qwen_scorer('cuda');sam,processor=load_sam3_model('cuda')
    rows=[]
    for question in questions:
        qid=question['question_id'];folder=args.output/qid;folder.mkdir(exist_ok=True)
        source=load_dataset_image(args.data_dir,question['image_id'])
        report=json.loads((args.batch/qid/'run/results.json').read_text())
        usage={}
        plan=qwen.resolve_structured(question['question'],
            on_request=lambda:charge_generation(usage,'resolver'))
        settings={**report['config'],'structured_query':True,'part_refinement':True,
            'query_plan':plan,'answer_view':'marked_detail','max_local_scores':4,
            'finalization_reserve':0,'max_part_detector_calls':1,'part_candidate_limit':3}
        def detect(pixels,prompt):
            return exec_sam3_on_image(pixels,sam,processor,prompt=prompt,threshold=.1,max_detections=8)
        state=RunState(query=report['query'],question=question['question'],settings=settings,
            usage=usage,source_image=source,rgb=np.asarray(source),artifacts=None,
            on_event=lambda *a:None,progress=lambda *a:None,custom_scorer=None,
            local_scorer=qwen,feature_extractor=lambda p:p,detector=detect)
        # Preserve the original top box and original verifier decision, even when wrong.
        result={'ranked_regions':[dict(p) for p in report['proposals'][:1]]}
        refine_answer_part(state,result)
        final=answer_top_region(state,result)
        if final:
            Image.fromarray(final.pop('input_pixels')).save(folder/'answer_input.png')
        row={**question,'query_plan':plan,'frozen_entity_bbox':report['proposals'][0]['bbox'] if report['proposals'] else None,
             'part_refinement':result.get('part_refinement'), 'answer':final,
             'abstention':result.get('abstention'),'usage':usage}
        (folder/'prediction.json').write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
        (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(qid, result.get('part_refinement',{}).get('status','skipped'),
              final['answer'] if final else 'Uncertain',usage,flush=True)


if __name__=='__main__':main()
