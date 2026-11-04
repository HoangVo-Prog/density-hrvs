import json
from types import SimpleNamespace
from types import SimpleNamespace
from algorithm.search.evaluation import candidate_query_text
import unittest

from algorithm.core.query_plan import parse_query_plan
from algorithm.search.setup import _resolve_sam_prompt


def badge_plan():
    return {'entity':{'concept':'man in dark polo shirt'},
            'attribute_region':{'concept':'badge','relation':'hanging from the lanyard of entity'},
            'requested_attribute':'color',
            'anchors':[{'id':'anchor_1','concept':'man in light blue shirt'}],
            'relations':[{'subject':'entity','predicate':'right_of','object':'anchor_1'}],
            'scene_constraints':['upper balcony']}


class QueryPlanTests(unittest.TestCase):
    def test_owner_stage_uses_full_relations_without_demanding_a_part_sized_box(self):
        question='What color is the badge of the man right of the woman in blue?'
        state=SimpleNamespace(settings={'structured_query':True,'query_plan':{
            'entity':{'concept':'man'},'attribute_region':{'concept':'badge'}}},
            question=question,query='original search description')
        query=candidate_query_text(state)
        self.assertIn(question,query)
        self.assertIn('owning/reference entity: man',query)
        self.assertIn('badge will be inspected separately',query)

    def test_entity_part_and_reference_remain_distinct(self):
        plan=parse_query_plan(json.dumps(badge_plan()))
        self.assertEqual(plan['entity']['concept'],'man in dark polo shirt')
        self.assertEqual(plan['attribute_region']['concept'],'badge')
        self.assertEqual(plan['anchors'][0]['concept'],'man in light blue shirt')

    def test_unknown_object_is_not_replaced_by_reference(self):
        plan={'entity':{'concept':'object hanging over railing'},'attribute_region':None,
              'requested_attribute':'identity','anchors':[{'id':'rail','concept':'upper right overpass railing'}],
              'relations':[{'subject':'entity','predicate':'hanging_over','object':'rail'}]}
        result=parse_query_plan(json.dumps(plan))
        self.assertEqual(result['entity']['concept'],'object hanging over railing')
        self.assertIsNone(result['attribute_region'])

    def test_invalid_reference_and_duplicate_identifier_are_rejected(self):
        plan=badge_plan();plan['relations'][0]['object']='missing'
        with self.assertRaisesRegex(ValueError,'unknown entity'):
            parse_query_plan(json.dumps(plan))
        plan=badge_plan();plan['anchors'][0]['id']='entity'
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            parse_query_plan(json.dumps(plan))

    def test_resolver_receives_original_question_and_records_generation(self):
        questions=[]
        def resolve(question,*,on_request):
            questions.append(question);on_request();return parse_query_plan(json.dumps(badge_plan()))
        settings={'structured_query':True,'sam_prompt':'badge'};usage={}
        _resolve_sam_prompt(settings,'stripped search query',SimpleNamespace(resolve_structured=resolve),
                            usage,lambda *args:None,'What color is the badge?')
        self.assertEqual(questions,['What color is the badge?'])
        self.assertEqual(settings['sam_prompt'],'man in dark polo shirt')
        self.assertEqual(usage['generation_counts'],{'resolver':1})

    def test_malformed_structured_result_uses_logged_legacy_fallback(self):
        events=[]
        scorer=SimpleNamespace(resolve_structured=lambda *a: (_ for _ in ()).throw(ValueError('bad plan')),
                               resolve=lambda q:'badge')
        settings={'structured_query':True,'sam_prompt':'badge'}
        _resolve_sam_prompt(settings,'query',scorer,{},lambda kind,*a:events.append(kind))
        self.assertEqual(settings['sam_prompt'],'badge')
        self.assertIn('query_resolution_failed',events)


if __name__=='__main__':unittest.main()
