import unittest

import numpy as np
from PIL import Image

from algorithm.search.parts import refine_answer_part
from algorithm.search.runtime import RunState


class PartRefinementTests(unittest.TestCase):
    def test_verified_part_is_linked_without_changing_primary_entity_box(self):
        source=Image.new('RGB',(200,200),'white');calls=[]
        def detector(pixels,prompt):
            calls.append(prompt)
            return [{'bbox':(40,40,50,50),'sam3_score':.8}]
        def scorer(image,query,*,view_type,on_request):
            on_request()
            return {'answer':'Yes','score':.95,'model_id':'test','reason':'badge belongs to magenta owner'}
        settings={'part_refinement':True,'structured_query':True,'verification_max_side':512,
                  'max_local_scores':8,'finalization_reserve':3,
                  'query_plan':{'entity':{'concept':'man'},'attribute_region':{'concept':'badge','relation':'on man'}}}
        state=RunState(query='badge of man',question='What color?',settings=settings,usage={},
            source_image=source,rgb=np.asarray(source),artifacts=None,on_event=lambda *a:None,
            progress=lambda *a:None,custom_scorer=None,local_scorer=scorer,
            feature_extractor=lambda p:p,detector=detector)
        owner={'bbox':(50,50,100,150),'candidate_verified':True,
               'verifier':{'answer':'Yes','score':.95,'model_id':'test'}}
        result={'ranked_regions':[owner]}
        refine_answer_part(state,result)
        self.assertEqual(calls,['badge'])
        self.assertEqual(owner['bbox'],(50,50,100,150))
        self.assertEqual(owner['attribute_bbox'],(58,58,68,68))
        self.assertEqual(result['part_refinement']['status'],'verified')
        self.assertEqual(state.usage['generation_counts'],{'part_verification':1})

    def test_empty_part_crop_retries_once_with_wider_context_and_correct_coordinates(self):
        source=Image.new('RGB',(200,200),'white');sizes=[]
        def detector(pixels,prompt):
            sizes.append(pixels.shape[:2])
            return [] if len(sizes)==1 else [{'bbox':(74,74,84,84),'sam3_score':.8}]
        def scorer(image,query,*,view_type,on_request):
            on_request()
            return {'answer':'Yes','score':.95,'model_id':'test','reason':'same owner'}
        settings={'part_refinement':True,'part_crop_retry':True,'max_part_detector_calls':2,
                  'max_local_scores':8,'verification_max_side':512,
                  'query_plan':{'entity':{'concept':'person'},'attribute_region':{'concept':'badge','relation':'on person'}}}
        state=RunState(query='badge on person',question='Color?',settings=settings,usage={},
            source_image=source,rgb=np.asarray(source),artifacts=None,on_event=lambda *a:None,
            progress=lambda *a:None,custom_scorer=None,local_scorer=scorer,
            feature_extractor=lambda p:p,detector=detector)
        owner={'bbox':(80,80,100,120),'candidate_verified':True,
               'verifier':{'answer':'Yes','score':.95,'model_id':'test'}}
        result={'ranked_regions':[owner]};refine_answer_part(state,result)
        self.assertEqual(sizes,[(104,84),(168,148)])
        self.assertEqual(owner['attribute_bbox'],(90,90,100,100))
        self.assertEqual(owner['bbox'],(80,80,100,120))
        self.assertEqual(state.usage['part_detector_calls'],2)
        self.assertEqual(len(result['part_refinement']['detector_attempts']),2)
        refine_answer_part(state,result)
        self.assertEqual(state.usage['part_detector_calls'],2)

    def test_no_part_plan_does_not_reserve_unused_part_generations(self):
        state=object.__new__(RunState)
        state.settings={'max_local_scores':5,'part_refinement':True,'finalization_reserve':2,
                        'query_plan':{'attribute_region':None}}
        state.usage={}
        self.assertEqual(state.tree_generation_limit(),5)
        for _ in range(5):state.charge_local()
        self.assertEqual(state.remaining_generations(),0)

    def test_postprocess_reserve_stays_inside_total_search_generation_limit(self):
        state=object.__new__(RunState)
        state.settings={'max_local_scores':5,'part_refinement':True,'finalization_reserve':2,
                        'query_plan':{'attribute_region':{'concept':'badge'}}}
        state.usage={}
        for _ in range(3):state.charge_local()
        self.assertEqual(state.remaining_generations(),0)
        self.assertEqual(state.remaining_final_generations(),2)
        state.charge_final('part_verification');state.charge_final('part_verification')
        self.assertEqual(state.remaining_final_generations(),0)
        self.assertEqual(state.usage['physical_generations'],5)


if __name__=='__main__':unittest.main()
