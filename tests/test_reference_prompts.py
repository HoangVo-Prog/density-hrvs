"""Reference extraction and multi-call routing regression tests (unit)."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from algorithm.core.reference_prompts import parse_reference_plan
from algorithm.search.setup import _resolve_sam_prompt
from algorithm.search.evaluation import detect_region, candidate_query_text
from algorithm.search.config import default_search_config, validate_search_config


class TestReferencePlan:
    def test_preserves_distinct_known_colors_and_deduplicates(self):
        plan = parse_reference_plan(json.dumps({'target':'cargo bed', 'references':[
            {'prompt':'turquoise car','context':'bottom approach'},
            {'prompt':'yellow bus','context':'left of car'},
            {'prompt':'Turquoise car','context':'duplicate'}]}))
        assert [r['prompt'] for r in plan['references']] == ['turquoise car','yellow bus']
        assert plan['target']=='cargo bed'

    def test_allows_no_reference_without_inventing_target_prompt(self):
        assert parse_reference_plan('{"target":"bus", "references":[]}')['references']==[]
        with pytest.raises(ValueError):
            parse_reference_plan('{"target":"car", "references":["bus"]}')


class TestReferenceRouting:
    def test_calls_each_reference_separately_and_caches_node(self):
        settings=default_search_config();settings.update(detector_prompt_mode='references')
        def resolve(question, *, on_request):
            on_request();return parse_reference_plan('{"target":"cargo bed", "references":[{"prompt":"turquoise car"},{"prompt":"yellow bus"}]}')
        usage={};_resolve_sam_prompt(settings,'cargo bed',SimpleNamespace(resolve_references=resolve),usage,lambda *a:None,'What color is the cargo bed?')
        calls=[]
        def detector(pixels,prompt):
            calls.append(prompt);return [{'bbox':(1,2,5,7),'sam3_score':.8}]
        state=SimpleNamespace(settings=settings,usage=usage,query='cargo bed',question='original question',detection_cache={},on_event=lambda *a:None,detector=detector,record_cost=lambda *a:None)
        node={'node_id':4,'bbox':(10,20,30,40)}
        found=detect_region(state,node,np.zeros((20,20,3),dtype=np.uint8))
        detect_region(state,node,np.zeros((20,20,3),dtype=np.uint8))
        assert calls==['turquoise car','yellow bus']
        assert usage['primary_detector_calls']==2 and usage['generation_counts']['resolver']==1
        assert [d['reference_id'] for d in found]==['reference_1','reference_2']
        assert settings['reference_detections'][0]['bbox']==[11,22,15,27]
        assert all(d['proposal_role']=='reference' for d in found)
        assert 'NOT the requested target' in candidate_query_text(state)

    def test_rejects_hidden_target_detector_fallback(self):
        settings=default_search_config();settings.update(detector_prompt_mode='references',grounding_dino_fallback=True)
        with pytest.raises(ValueError,match='target/part'):
            validate_search_config(settings)


class TestReferenceSuppression:
    def test_keeps_overlapping_different_references(self):
        from algorithm.search.region_scoring import distinct_detections
        config=default_search_config();config['detector_prompt_mode']='references'
        boxes=[{'bbox':(0,0,10,10),'sam3_score':.9,'reference_id':'reference_1'},
               {'bbox':(0,0,10,10),'sam3_score':.8,'reference_id':'reference_2'},
               {'bbox':(0,0,10,10),'sam3_score':.7,'reference_id':'reference_1'}]
        assert len(distinct_detections(boxes,config))==2
