"""Bounded, visually verified attribute-region refinement of the chosen entity.

The primary proposal/IoU convention stays entity-level. Attribute boxes are
separate linked outputs and are never selected using evaluation annotations.
"""
from __future__ import annotations

import json
import time

from PIL import Image

from ..core.errors import InvalidScoreResponse, LocalBudgetExceeded
from ..models.evidence import build_proposal_evidence, evidence_info
from .eligibility import eligible_regions
from .answering import padded_bbox
from .recovery import fallback_boxes


def refine_answer_part(state, result) -> None:
    if not state.settings.get('part_refinement', False):
        return
    plan = state.settings.get('query_plan', {})
    part = plan.get('attribute_region')
    eligible = eligible_regions(result['ranked_regions'], state.settings)
    if not part or not eligible:
        return
    best = eligible[0]
    if state.remaining_final_generations() == 0:
        result["part_refinement"] = {"status": "generation_budget"}
        return
    bbox = best['bbox']
    padding = max(32, round(max(bbox[2]-bbox[0], bbox[3]-bbox[1])*.25))
    window = padded_bbox(bbox, state.source_image.size, padding)
    x0,y0,x1,y1 = window
    limit = state.settings.get('max_part_detector_calls', 1)
    used = state.usage.get('part_detector_calls', 0)
    if used >= limit:
        result['part_refinement'] = {'status':'detector_budget', 'entity_bbox':bbox}
        return
    folder = state.artifacts/'parts' if state.artifacts is not None else None
    if folder is not None: folder.mkdir(parents=True, exist_ok=True)
    old_folder, old_node = state.context.folder, state.context.node_id
    tick = time.perf_counter()
    state.usage['part_detector_calls'] = used + 1
    state.context.folder, state.context.node_id = folder, -1
    try:
        detections = state.detector(state.rgb[y0:y1,x0:x1].copy(), part['concept'])
    finally:
        state.context.folder, state.context.node_id = old_folder, old_node
        state.record_cost('detect', time.perf_counter()-tick)
    if not detections:
        detections = fallback_boxes(state, state.rgb[y0:y1,x0:x1].copy(), part["concept"],
            reason="empty_requested_part", source_bbox=window)
    records=[]
    candidates=sorted(detections,key=lambda d:d.get('sam3_score') or 0,reverse=True)
    for index,detection in enumerate(candidates[:state.settings.get('part_candidate_limit',3)],1):
        if state.remaining_final_generations()==0:break
        a,b,c,d=detection['bbox']
        box=(max(0,x0+a),max(0,y0+b),min(state.rgb.shape[1],x0+c),min(state.rgb.shape[0],y0+d))
        if box[0]>=box[2] or box[1]>=box[3]:continue
        evidence=build_proposal_evidence(state.source_image,box,
            max_side=state.settings.get('verification_max_side',1536),
            context_padding=max(state.settings.get('context_padding',384), max(bbox[2]-bbox[0],bbox[3]-bbox[1])),
            reference_bbox=bbox,
            detail_padding=state.settings.get('detail_padding',16))
        query=(f"The CYAN-boxed target must be the {part['concept']} associated with the SAME selected "
               f"{plan['entity']['concept']}. Required ownership/relation: {part['relation']}. "
               f"The MAGENTA rectangle identifies the selected owner/reference. Original search description: {state.query}. Verify the cyan part belongs to that magenta owner, not to a nearby person.")
        try:
            rating=state.local_scorer(Image.fromarray(evidence['image']),query,
                view_type='context_detail',on_request=lambda:state.charge_final('part_verification'))
        except (LocalBudgetExceeded,InvalidScoreResponse) as exc:
            records.append({'bbox':box,'status':type(exc).__name__})
            break
        record={'bbox':box,'concept':part['concept'],'owner_bbox':bbox,
                'sam3_score':detection.get('sam3_score'),'verifier':rating,
                'evidence_info':evidence_info(evidence)}
        records.append(record)
        if folder is not None:
            Image.fromarray(evidence['image']).save(folder/f'candidate_{index:03d}.png')
        if rating['answer']=='Yes' and rating['score']>=state.settings.get('candidate_accept_score',.5):
            best['entity_bbox']=bbox
            best['attribute_bbox']=box
            best['attribute_verifier']=rating
            best['attribute_concept']=part['concept']
            best['role']='entity'
            # The primary proposal bbox stays fixed; only answer detail uses the part.
            break
    result['part_refinement']={'status':'verified' if best.get('attribute_bbox') else 'unresolved',
                               'entity_bbox':bbox,'concept':part['concept'],'candidates':records}
    if folder is not None:
        (folder/'results.json').write_text(json.dumps(result['part_refinement'],indent=2)+'\n')
