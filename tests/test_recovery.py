import unittest
from types import SimpleNamespace
import numpy as np

from algorithm.search.recovery import fallback_boxes, recover_empty_region
from algorithm.search.evaluation import detect_region


class RecoveryTests(unittest.TestCase):
    def state(self):
        settings={'grounding_dino_fallback':True,'max_fallback_detector_calls':1,
            'sam_prompt':'scooter','sam3_threshold':.1,'max_proposals_per_node':8}
        calls=[]
        def backup(pixels,prompt):
            calls.append(prompt)
            return [{'bbox':(1,1,4,4),'sam3_score':None,'proposal_backend':'test',
                     'detector_score':.6,'mask_kind':'box_only'}]
        state=SimpleNamespace(settings=settings,usage={},artifacts=None,recovery_detector=backup,
            remaining_final_generations=lambda:4,observed_clusters=[],detection_cache={},
            on_event=lambda *a:None,record_cost=lambda *a:None,query='scooter',
            detector=lambda *a:[])
        return state,calls

    def test_root_miss_can_recover_but_unpromising_tiles_cannot(self):
        state,calls=self.state();pixels=np.zeros((10,10,3),dtype=np.uint8)
        self.assertEqual(recover_empty_region(state,{'node_id':1,'parent_id':0,'bbox':(0,0,10,10)},pixels),[])
        boxes=recover_empty_region(state,{'node_id':0,'bbox':(0,0,10,10)},pixels)
        self.assertEqual(len(boxes),1);self.assertIsNone(boxes[0]['sam3_score'])
        self.assertEqual(calls,['scooter'])
        self.assertEqual(recover_empty_region(state,{'node_id':0,'bbox':(0,0,10,10)},pixels),[])
        self.assertEqual(state.usage['fallback_detector_calls'],1)

    def test_recovered_results_are_cached_and_do_not_repeat_model_calls(self):
        state,calls=self.state();node={'node_id':0,'bbox':(0,0,10,10)}
        pixels=np.zeros((10,10,3),dtype=np.uint8)
        self.assertEqual(len(detect_region(state,node,pixels)),1)
        self.assertEqual(len(detect_region(state,node,pixels)),1)
        self.assertEqual(calls,['scooter'])
        self.assertEqual(state.usage['primary_detector_calls'],1)

    def test_no_remaining_verifier_budget_prevents_new_detector_work(self):
        state,calls=self.state();state.remaining_final_generations=lambda:0
        boxes=fallback_boxes(state,np.zeros((10,10,3),dtype=np.uint8),'badge',
            reason='empty_requested_part',source_bbox=(0,0,10,10))
        self.assertEqual(boxes,[]);self.assertEqual(calls,[])
