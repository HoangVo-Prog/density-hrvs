import unittest
from types import SimpleNamespace
from PIL import Image

from algorithm.core.calls import charge_generation
from algorithm.search.eligibility import eligible_regions
from algorithm.search.reranking import group_candidates, shortlist_candidates, rerank_regions
from algorithm.models.comparison import build_comparison_sheet


class RerankingTests(unittest.TestCase):
    def test_duplicates_preserve_part_owner_and_adjacent_instances(self):
        pool = [{'bbox': (0,0,100,100), 'role':'entity'},
                {'bbox': (1,1,101,101), 'role':'entity'},
                {'bbox': (0,0,100,100), 'role':'attribute'},
                {'bbox': (20,0,120,100), 'role':'entity'},
                {'bbox': (10,10,30,30), 'role':'entity'}]
        groups = group_candidates(pool)
        self.assertEqual([g['members'] for g in groups], [[0,1],[2],[3],[4]])
        self.assertEqual(shortlist_candidates(pool, groups)[0], 0)
        self.assertEqual(sum(len(g['members']) for g in groups),len(pool))

    def test_comparative_rejection_cannot_fall_back_to_isolated_yes(self):
        pool=[{'bbox':(10,10,40,40),'verifier':{'answer':'Yes','score':.95,'model_id':'old'}}]
        def compare(image,query,count,*,on_request):
            on_request()
            return {'ranking':[1],'best_index':1,'ambiguous':True,
                    'ratings':{1:{'answer':'Uncertain','score':.5,'model_id':'test'}}}
        scorer=lambda *a:None;scorer.compare_candidates=compare
        usage={};settings={'comparative_reranking':True}
        state=SimpleNamespace(settings=settings,source_image=Image.new('RGB',(100,100)),
            artifacts=None,query='person',local_scorer=scorer,remaining_final_generations=lambda:2,
            charge_final=lambda stage:charge_generation(usage,stage,2),record_cost=lambda *a:None)
        result={'ranked_regions':pool};rerank_regions(state,result)
        self.assertEqual(result['reranking']['status'],'no_unambiguous_match')
        self.assertEqual(eligible_regions(pool,settings),[])
        self.assertEqual(usage['physical_generations'],1)
        self.assertEqual(len(result['ranked_regions']),1)

    def test_selected_candidate_preserves_pool_and_verified_identity(self):
        pool=[{'bbox':(10,10,40,40),'verifier':{'answer':'Yes','score':.95,'model_id':'old'}},
              {'bbox':(60,60,90,90),'verifier':{'answer':'No','score':.1,'model_id':'old'}}]
        def compare(image,query,count,*,on_request):
            on_request()
            return {'ranking':[2,1],'best_index':2,'ambiguous':False,
                    'ratings':{1:{'answer':'No','score':.1,'model_id':'test'},
                               2:{'answer':'Yes','score':.9,'model_id':'test'}}}
        scorer=lambda *a:None;scorer.compare_candidates=compare
        state=SimpleNamespace(settings={'comparative_reranking':True},source_image=Image.new('RGB',(100,100)),
            artifacts=None,query='person',local_scorer=scorer,remaining_final_generations=lambda:2,
            charge_final=lambda stage:None,record_cost=lambda *a:None)
        result={'ranked_regions':pool};rerank_regions(state,result)
        self.assertIs(result['ranked_regions'][0], pool[1])
        self.assertEqual(result['ranked_regions'][0]['previous_verifier']['answer'],'No')
        self.assertEqual(len(result['ranked_regions']),2)
        self.assertEqual(len(eligible_regions(result['ranked_regions'],state.settings)),1)

    def test_comparison_mapping_uses_original_pixels_and_bounded_tiles(self):
        source=Image.new('RGB',(200,200),'red')
        sheet,mapping=build_comparison_sheet(source,[{'bbox':(20,20,50,50)}],512)
        self.assertEqual(sheet.size,(512,341))
        self.assertEqual(mapping[0]['source_bbox'],(20,20,50,50))
        with self.assertRaises(ValueError):build_comparison_sheet(source,[],512)
