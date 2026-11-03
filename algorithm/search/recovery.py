"""Bounded empty-crop recovery; root or inference-promising regions only."""
import json
import time

from ..models.grounding_dino import detect_grounding_dino


def fallback_boxes(state, pixels, prompt, *, reason, source_bbox):
    if not state.settings.get('grounding_dino_fallback', False):return []
    used=state.usage.get('fallback_detector_calls',0)
    if used >= state.settings.get('max_fallback_detector_calls',1):return []
    if state.remaining_final_generations()==0:return []
    state.usage['fallback_detector_calls']=used+1
    audit={'backend':'IDEA-Research/grounding-dino-tiny','reason':reason,
           'source_bbox':source_bbox,'prompt':prompt}
    tick=time.perf_counter()
    def capture(records):audit['raw_candidates']=records
    try:
        provided=getattr(state,'recovery_detector',None)
        if provided is not None:
            boxes=provided(pixels,prompt)
        else:
            boxes=detect_grounding_dino(pixels,prompt,device=state.settings['device'],
                cache_dir=state.settings.get('fallback_cache_dir'),
                threshold=state.settings.get('fallback_threshold',.25),
                text_threshold=state.settings.get('fallback_text_threshold',.25),
                limit=state.settings.get('fallback_proposal_limit',8),on_candidates=capture)
        audit['retained_boxes']=[{'bbox':d['bbox'],'confidence':d.get('detector_score')} for d in boxes]
        return boxes
    finally:
        audit['elapsed_seconds']=time.perf_counter()-tick
        state.usage.setdefault('fallback_diagnostics',[]).append(audit)
        if state.artifacts is not None:
            folder=state.artifacts/'recovery';folder.mkdir(parents=True,exist_ok=True)
            (folder/f'fallback_{used+1:02d}.json').write_text(json.dumps(audit,indent=2)+'\n')


def recover_empty_region(state,node,pixels):
    if not state.settings.get('grounding_dino_fallback',False):return []
    # Root is a global detector miss. Descendants need prior positive visual evidence.
    parent=next((p for p in state.observed_clusters if p['cluster_id']==node.get('parent_id')),None)
    promising=node['node_id']==0 or (parent is not None and parent['verifier']['answer']=='Yes'
        and parent['verifier']['score'] >= state.settings.get('recovery_promising_score',.7))
    if not promising:return []
    return fallback_boxes(state,pixels,state.settings['sam_prompt'] or state.query,
        reason='empty_root' if node['node_id']==0 else 'empty_promising_child',source_bbox=node['bbox'])
