"""Question-region search must verify node0 before any partition or child call."""
import numpy as np
from algorithm.search.tree import explore_tree
from algorithm.search.config import default_search_config


def run_case(root_score):
    events=[];cfg=default_search_config();cfg.update(verification_mode='question_region',max_iterations=3,max_depth=2,stop_on_match=True,match_threshold=.9)
    root={'bbox':(0,0,100,100),'mask':np.ones((100,100),bool)}
    def evaluate(n):
        events.append(('evaluate',n['node_id']))
        score=root_score if n['node_id']==0 else 1.
        return {'assessment':{'answer':'Yes' if score>=.9 else 'No','score':score},'raw_reward':score,
            'proposals':[{'score':score,'candidate_verified':score>=.9,'verifier':{'answer':'Yes' if score>=.9 else 'No'}}]}
    def split(n):
        events.append(('split',n['node_id']))
        return [{'bbox':(0,0,50,100),'mask':np.ones((100,50),bool)}]
    result=explore_tree(root,'question',cfg,evaluate,split,lambda n:1.,{'api_calls':0},progress=lambda *a:None)
    return result,events


def test_solved_root_never_partitions_or_runs_child_detection():
    result,events=run_case(1.)
    assert events==[('evaluate',0)]
    assert result['stop_reason']=='match' and len(result['nodes'])==1
    assert result['trace'][0]['path']==[0]


def test_unsolved_root_partitions_only_after_its_verification():
    result,events=run_case(.3)
    assert events==[('evaluate',0),('split',0),('evaluate',1)]
    assert [s['path'] for s in result['trace']]==[[0],[0,1]]
    assert result['nodes'][0]['evaluated'] and result['nodes'][1]['evaluated']


def test_child_refreshes_every_object_prompt_instead_of_reusing_root_boxes():
    from functools import partial
    from algorithm.search.pipeline import evaluate_node
    from test_question_region import state_with, node
    detections=[];checks=[]
    def detector(pixels,prompt):
        detections.append((pixels.shape[:2],prompt))
        return [{"bbox":[2,2,8,8],"sam3_score":.8}]
    def verifier(im,q,*,spatial_context,on_request):
        on_request();checks.append(spatial_context['metadata']['node_id'])
        score=.2 if len(checks)==1 else 1.
        return {'answer':'No' if len(checks)==1 else 'Yes','score':score,
            'score_source':'yes_no_token_probability','model_id':'mock'}
    state=state_with(detector,verifier);state.usage['api_calls']=0
    state.settings.update(region_spatial_context=True,max_iterations=3,match_threshold=.9,stop_on_match=True)
    root={'bbox':(0,0,50,40),'mask':np.ones((40,50),bool)}
    result=explore_tree(root,'question',state.settings,partial(evaluate_node,state),
        lambda n:[{'bbox':node()['bbox'],'mask':node()['mask']}],lambda n:1.,state.usage,progress=lambda *a:None)
    assert checks==[0,1]
    assert detections==[((40,50),'white structure'),((40,50),'green area'),((20,20),'white structure'),((20,20),'green area')]
    assert set(state.detection_cache)=={0,1}
    assert state.usage['primary_detector_calls']==4
    assert result['stop_reason']=='match'
