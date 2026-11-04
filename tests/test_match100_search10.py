"""Actual tree stopping behavior for verified 1.0 matches and a 10-visit/search-generation cap."""
import numpy as np
from algorithm.search.config import default_search_config
from algorithm.search.tree import explore_tree
from algorithm.search.partitioning import spatial_children
from algorithm.core.calls import charge_generation


def search(score,*,verified=True,generations_per_node=1):
    config=default_search_config();config.update(stop_on_match=True,match_threshold=1.0,max_iterations=10,max_local_scores=10)
    root={'bbox':(0,0,128,128),'mask':np.ones((128,128),dtype=bool)}
    usage={'api_calls':0}
    def evaluate(n):
        for _ in range(generations_per_node):charge_generation(usage,'verification',10)
        rating={'answer':'Yes' if verified else 'Uncertain','score':score,'reason':'test evidence','model_id':'mock'}
        return {'assessment':rating,'raw_reward':score,'proposals':[{'score':score,'candidate_verified':verified,'verifier':rating}]}
    return explore_tree(root,'question',config,evaluate,spatial_children,lambda n:1.,usage,progress=lambda *a:None)


class TestMatch100Search10:
    def test_stops_on_first_verified_one_point_zero_match(self):
        result=search(1.)
        assert result['stop_reason']=='match' and len(result['trace'])==1
        assert result['usage']['search_generations']==1

    def test_point_nine_nine_continues_but_never_exceeds_ten_visits(self):
        result=search(.99)
        assert result['stop_reason']=='iterations' and len(result['trace'])==10
        assert result['usage']['search_generations']==10

    def test_unverified_one_point_zero_does_not_trigger_early_stop(self):
        result=search(1.,verified=False)
        assert result['stop_reason']=='iterations' and len(result['trace'])==10

    def test_repair_generations_also_count_against_ten_generation_limit(self):
        result=search(.8,generations_per_node=2)
        assert result['stop_reason']=='local_budget'
        assert result['usage']['search_generations']==10
        assert sum(t['completed'] for t in result['trace'])==5
