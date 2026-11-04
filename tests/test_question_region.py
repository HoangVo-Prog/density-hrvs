"""Original-question region verification and independent object-prompt routing tests."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from algorithm.core.reference_prompts import parse_object_plan
from algorithm.search.config import default_search_config,validate_search_config
from algorithm.search.setup import _resolve_sam_prompt
from algorithm.search.runtime import RunState
from algorithm.search.pipeline import evaluate_node
from algorithm.search.eligibility import answer_eligible
from algorithm.search.answering import answer_top_region

QUESTION='What shape is the white structure in the green area at the bottom-left of the intersection?'


def settings():
    c=default_search_config();c.update(detector_prompt_mode='objects_and_references',verification_mode='question_region',sam3_threshold=.25,sam3_fallback_top_k=10,answer_context_padding=0)
    c['reference_plan']=parse_object_plan(json.dumps({'target':'white structure','objects':[
        {'role':'target','prompt':'white structure','context':'bottom-left green area'},
        {'role':'reference','prompt':'green area','context':'bottom-left'}]}));c['reference_detections']=[]
    return c


def state_with(detector,verifier,answerer=None):
    rgb=np.zeros((40,50,3),dtype=np.uint8);rgb[10:30,10:30]=[42,127,210]
    scorer=SimpleNamespace(score_question_region=verifier,answer=answerer,model_id='mock',model_revision='test',quantization='none')
    return RunState(query='the white structure',question=QUESTION,settings=settings(),usage={'cache_hits':0},source_image=Image.fromarray(rgb),rgb=rgb,
        artifacts=None,on_event=lambda *a:None,progress=lambda *a:None,custom_scorer=None,local_scorer=scorer,
        feature_extractor=lambda pixels:np.zeros((*pixels.shape[:2],3)),detector=detector)


def node():
    return {'node_id':3,'bbox':(10,10,30,30),'mask':np.ones((20,20),dtype=bool),'depth':1}


class TestObjectPlan:
    def test_target_is_mandatory_and_reference_is_distinct(self):
        plan=settings()['reference_plan']
        assert [(o['role'],o['prompt']) for o in plan['objects']]==[('target','white structure'),('reference','green area')]
        with pytest.raises(ValueError,match='omitted'):
            parse_object_plan('{"target":"white structure","objects":[{"role":"reference","prompt":"green area"}]}')

    def test_resolver_receives_original_question_without_reference_only_rewrite(self):
        c=settings();calls=[]
        def resolve(q,*,on_request):
            assert q==QUESTION;on_request();calls.append(q);return c['reference_plan']
        usage={};_resolve_sam_prompt(c,'the white structure',SimpleNamespace(resolve_objects=resolve),usage,lambda *a:None,QUESTION)
        assert len(calls)==1 and usage['generation_counts']['resolver']==1
        assert c['original_question']==QUESTION
        assert 'white structure' in c['sam_prompt'] and 'green area' in c['sam_prompt']


class TestQuestionRegion:
    def test_verifies_clean_region_even_when_every_detector_prompt_is_empty(self):
        detector_calls=[];seen=[]
        def detector(pixels,prompt):detector_calls.append(prompt);return []
        def verifier(im,q,*,on_request):
            assert q==QUESTION and im.size==(20,20)
            assert np.all(np.asarray(im)==[42,127,210])  # no colored boxes or context/detail sheet
            on_request();seen.append(q)
            return {'answer':'Yes','score':.95,'reason':'Requested structure visible','model_id':'mock'}
        s=state_with(detector,verifier);e=evaluate_node(s,node())
        assert detector_calls==['white structure','green area'] and seen==[QUESTION]
        assert e['raw_reward']==.95 and answer_eligible(e['proposals'][0],s.settings)
        assert e['proposals'][0]['bbox']==node()['bbox'] and e['proposals'][0]['proposal_role']=='answer_region'
        assert e['proposals'][0]['sam3_score'] is None
        assert s.usage['question_region_checks']==1 and s.usage['search_generations']==1
        assert s.usage.get('gated_regions',0)==0

    def test_reference_confidence_cannot_make_an_unanswerable_region_eligible(self):
        def detector(pixels,prompt):return [{'bbox':(0,0,10,10),'sam3_score':.999,'sam3_selection':'above_threshold'}]
        def verifier(im,q,*,on_request):
            on_request();return {'answer':'No','score':0.,'reason':'White structure absent','model_id':'mock'}
        s=state_with(detector,verifier);e=evaluate_node(s,node())
        assert e['raw_reward']==0 and not answer_eligible(e['proposals'][0],s.settings)
        assert len(s.settings['reference_detections'])==2
        assert {b['proposal_role'] for b in s.settings['reference_detections']}=={'target','reference'}

    def test_each_prompt_retains_its_own_threshold_or_fallback_results(self):
        def detector(pixels,prompt):
            if prompt=='white structure':return [{'bbox':(0,0,10,10),'sam3_score':.9,'sam3_selection':'above_threshold'}]
            return [{'bbox':(i,0,i+2,5),'sam3_score':.01-i*.0001,'sam3_selection':'top_k_fallback'} for i in range(10)]
        def verifier(im,q,*,on_request):on_request();return {'answer':'Yes','score':.8,'reason':'Visible','model_id':'mock'}
        s=state_with(detector,verifier);evaluate_node(s,node());boxes=s.settings['reference_detections']
        assert len(boxes)==11
        assert len([b for b in boxes if b['proposal_role']=='target'])==1
        assert len([b for b in boxes if b['proposal_role']=='reference' and b['sam3_selection']=='top_k_fallback'])==10
        assert s.usage['sam3_top_k_fallback_calls']==1

    def test_answer_uses_original_question_and_verified_region_without_reference_crop(self):
        def detector(*a):return []
        def verifier(im,q,*,on_request):on_request();return {'answer':'Yes','score':.9,'reason':'Visible','model_id':'mock'}
        def answerer(im,q,*,on_request):
            assert q==QUESTION and im.size==(20,20)
            on_request();return 'crescent-shaped'
        s=state_with(detector,verifier,answerer);evaluate_node(s,node())
        result={'ranked_regions':s.observed_regions};answer=answer_top_region(s,result)
        assert answer['answer']=='crescent-shaped' and answer['crop_bbox']==node()['bbox']
