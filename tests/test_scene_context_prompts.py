"""Prevent scene-position fallback boxes from being treated as reference objects."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from algorithm.core.reference_prompts import parse_object_plan, is_scene_position_prompt
from algorithm.search.evaluation import detect_region
from algorithm.search.config import default_search_config


@pytest.mark.parametrize("text", ["right-hand approach", "bottom approach", "uppermost queue", "lowest full queue", "right side of the intersection"])
def test_scene_position_is_not_an_object(text):
    assert is_scene_position_prompt(text)


@pytest.mark.parametrize("text", ["blue bus", "white-cab pickup", "pickup cargo bed", "traffic light", "crosswalk", "green area"])
def test_physical_reference_is_not_removed_by_position_guard(text):
    assert not is_scene_position_prompt(text)


def test_only_physical_target_and_reference_prompts_are_sent_to_sam3():
    plan=parse_object_plan(json.dumps({'target':'cargo bed','objects':[
        {'role':'target','prompt':'cargo bed','context':'above-right of blue bus'},
        {'role':'reference','prompt':'blue bus','context':'right-hand approach'},
        {'role':'reference','prompt':'right-hand approach','context':'location'}]}))
    assert [o['prompt'] for o in plan['objects']]==['cargo bed','blue bus']
    assert plan['context_only'][0]['prompt']=='right-hand approach'
    assert plan['objects'][1]['context']=='right-hand approach'
    config=default_search_config();config.update(detector_prompt_mode='objects_and_references',reference_plan=plan,reference_detections=[])
    calls=[]
    def detector(pixels,prompt):
        calls.append(prompt);return [{'bbox':[0,0,5,5],'sam3_score':.9}]
    state=SimpleNamespace(settings=config,usage={},query='cargo bed',question='original',detection_cache={},on_event=lambda *a:None,record_cost=lambda *a:None,detector=detector)
    detect_region(state,{'node_id':2,'bbox':[10,10,30,30]},np.zeros((20,20,3),dtype=np.uint8))
    assert calls==['cargo bed','blue bus']


def test_location_cannot_replace_the_requested_target():
    with pytest.raises(ValueError,match='physical object'):
        parse_object_plan('{"target":"right side","objects":[{"role":"target","prompt":"right side"}]}')
