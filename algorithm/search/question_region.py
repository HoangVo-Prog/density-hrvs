"""Verify the original question on the whole clean region, independently of SAM3 boxes."""
import json
import time
import numpy as np
from ..core.calls import invoke_counted
from ..core.imaging import to_pil_rgb
from ..reporting.artifacts import save_node_detections
from .evaluation import detect_region
from .region_scoring import distinct_detections


def evaluate_question_region(state,node,pixels,folder):
    # Proposals are target/reference localization aids. Their absence never gates Qwen.
    detections=detect_region(state,node,pixels)
    distinct=distinct_detections(detections,state.settings)
    image=to_pil_rgb(pixels)
    key=state.view_key(image,"region",state.question)
    if key in state.score_cache:
        state.usage["cache_hits"]+=1;rating=state.score_cache[key]
    else:
        verifier=getattr(state.local_scorer,"score_question_region",None)
        if not callable(verifier):raise ValueError("Question-region mode requires Qwen score_question_region")
        tick=time.perf_counter()
        state.usage["logical_score_requests"]=state.usage.get("logical_score_requests",0)+1
        try:rating=invoke_counted(verifier,image,state.question,on_request=state.charge_local)
        finally:state.record_cost("score",time.perf_counter()-tick)
        state.score_cache[key]=rating
    quality=rating["score"] if rating["answer"]!="No" else 0.
    verified=rating["answer"]=="Yes" and rating["score"]>=state.settings.get("candidate_accept_score",.5)
    state.usage["question_region_checks"]=state.usage.get("question_region_checks",0)+1
    components={"qwen_score":rating["score"],"raw_box_count":len(detections),"unique_box_count":len(distinct),
        "mean_confidence":0.,"count_score":0.,"mean_pairwise_iou":0.,"diversity_score":0.,
        "combined_score":quality,"gate":"question_region","verified_box_count":0,"candidate_checks":0}
    # The verified unit is the REGION. Never transfer its rating to a reference box.
    h,w=pixels.shape[:2]
    proposal={"bbox":node["bbox"],"local_bbox":(0,0,w,h),"mask":np.asarray(node["mask"]).copy(),
        "image":pixels.copy(),"score":quality,"sam3_score":None,"cluster_id":node["node_id"],
        "cluster_score":quality,"verifier":rating,"candidate_rank":1,"candidate_verified":verified,
        "scoring_scope":"region","score_components":components,"proposal_backend":"qwen:question_region",
        "proposal_role":"answer_region","scoring_method":"qwen3vl_original_question_clean_region"}
    from .pipeline import _record_cluster,register_proposals
    _record_cluster(state,node,pixels,quality,rating,components)
    register_proposals(state,[proposal])
    save_node_detections(pixels,detections,distinct,components,folder)
    if folder is not None:
        image.save(folder/"question_region_input.png")
        (folder/"question_region_score.json").write_text(json.dumps({"question":state.question,"rating":rating},indent=2))
    state.on_event("question_region_verified","Qwen evaluated original question on the clean region.",
        {"node_id":node["node_id"],"question":state.question,"qwen":rating,"detector_boxes":len(detections)})
    return {"assessment":rating,"proposals":[proposal],"raw_reward":quality,"candidate_checks":[]}
