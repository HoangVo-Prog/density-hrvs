import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from algorithm.search.candidates import verify_region_candidates
from algorithm.search.eligibility import eligible_regions
from algorithm.search.pipeline import evaluate_candidates
from algorithm.core.errors import LocalBudgetExceeded


def proposal(answer, verified=True, score=.95):
    return {'verifier': {'answer': answer, 'score': score, 'model_id': 'test'},
            'candidate_verified': verified}


class CandidateSafetyTests(unittest.TestCase):
    def test_rejected_or_unchecked_top_does_not_hide_verified_runner_up(self):
        good = proposal('Yes')
        self.assertEqual(eligible_regions([proposal('No'), proposal('Uncertain', False), good], {}), [good])
        self.assertEqual(eligible_regions([proposal('No')], {}), [])

    def test_safety_switch_preserves_explicit_legacy_comparison(self):
        bad = proposal('No')
        self.assertEqual(eligible_regions([bad], {'require_verified_answer': False}), [bad])

    def test_five_tied_candidates_respect_fallback_and_remaining_budget(self):
        for fallback, remaining, expected in [(0, 20, 0), (1, 20, 1), (3, 20, 3), (3, 1, 1)]:
            with self.subTest(fallback=fallback, remaining=remaining):
                ratings = {i: {'answer': 'Yes', 'score': .95, 'reason': 'tie', 'model_id': 'test'} for i in range(1, 6)}
                assessment = {'ratings': ratings, 'ranking': list(ratings), 'best_index': 1, 'ambiguous': True}
                scorer = SimpleNamespace(score_candidates=lambda *a, **kw: assessment)
                state = SimpleNamespace(local_scorer=scorer, settings={
                    'region_candidate_limit': 5, 'region_fallback_limit': fallback,
                    'sam_prompt': 'person', 'candidate_qwen_weight': .7, 'candidate_sam3_weight': .3},
                    query='person', usage={}, checked_cache={}, rgb=np.zeros((16,16,3), dtype=np.uint8), on_event=lambda *a: None,
                    record_cost=lambda *a: None, remaining_generations=lambda: remaining,
                    charge_local=lambda: None)
                detections = [{'bbox': (i, i, i+3, i+3), 'sam3_score': .9} for i in range(5)]
                node = {'node_id': 1, 'bbox': (0, 0, 16, 16)}
                indices = []
                def inspect(state, node, distinct, chosen, folder):
                    indices.extend(chosen)
                    return []
                with patch('algorithm.search.candidates.build_region_candidate_sheet', return_value=Image.new('RGB', (16,16))), patch('algorithm.search.candidates.verify_selected_candidates', side_effect=inspect):
                    verify_region_candidates(state, node, detections, np.zeros((16,16,3), dtype=np.uint8), None)
                self.assertEqual(len(indices), expected)

    def test_budget_interruption_retains_checked_proposal_without_reward_backup(self):
        detection = {'bbox': (0,0,4,4), 'sam3_score': .9}
        check = {'detection_index': 0, 'candidate_score': .9, 'qwen': {
            'answer': 'Yes', 'score': .95, 'model_id': 'test', 'reason': 'visible'}}
        state = SimpleNamespace(settings={'verification_mode': 'region'}, usage={}, checked_cache={1:[check]}, on_event=lambda *a: None)
        node = {'node_id': 1, 'bbox': (0,0,8,8)}
        with patch('algorithm.search.pipeline.detect_region', return_value=[detection]), patch('algorithm.search.pipeline.distinct_detections', return_value=[detection]), patch('algorithm.search.pipeline.verify_region_candidates', side_effect=LocalBudgetExceeded('spent')), patch('algorithm.search.pipeline._candidate_proposals', return_value=[{'id': 'saved'}]), patch('algorithm.search.pipeline.register_proposals') as register, patch('algorithm.search.pipeline.save_node_detections'):
            with self.assertRaises(LocalBudgetExceeded):
                evaluate_candidates(state, node, np.zeros((8,8,3), dtype=np.uint8), None)
        register.assert_called_once_with(state, [{'id':'saved'}])


if __name__ == '__main__':
    unittest.main()
