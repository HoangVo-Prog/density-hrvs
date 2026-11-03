import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

from algorithm.models.evidence import build_answer_evidence, build_proposal_evidence
from algorithm.reporting.artifacts import save_search_result
from algorithm.search.answering import answer_top_region


class AnswerEvidenceTests(unittest.TestCase):
    def test_detail_preserves_color_pixels_without_cyan_marker(self):
        original = Image.new('RGB', (160, 120), (160, 60, 80))
        clean = build_answer_evidence(original, (40, 30, 80, 70), max_side=512,
                                      context_padding=20, detail_padding=4)
        marked = build_proposal_evidence(original, (40, 30, 80, 70), max_side=512,
                                         context_padding=20, detail_padding=4)
        cyan = np.array([0, 217, 255])
        boundary = int(512 * .64)
        self.assertTrue(np.any(np.all(clean['image'][:, :boundary] == cyan, axis=2)))
        self.assertFalse(np.any(np.all(clean['image'][:, boundary:] == cyan, axis=2)))
        self.assertTrue(np.any(np.all(marked['image'][:, boundary:] == cyan, axis=2)))

    def test_saved_answer_image_is_the_actual_model_input(self):
        seen = []
        def answer(image, question, *, on_request):
            on_request()
            seen.append(np.asarray(image).copy())
            return 'red'
        source = Image.new('RGB', (160, 120), 'red')
        proposal = {'bbox': (40,30,80,70), 'score': .95,
                    'candidate_verified': True, 'candidate_rank': 2, 'cluster_id': 7,
                    'verifier': {'answer':'Yes','score':.95,'model_id':'test'}}
        usage = {}
        state = SimpleNamespace(settings={'answer_view':'marked_detail','verification_max_side':512},
            source_image=source, question='What color?', usage=usage,
            local_scorer=SimpleNamespace(answer=answer, answer_evidence=answer),
            charge_answer=lambda: usage.update(answer_generations=usage.get('answer_generations',0)+1))
        final = answer_top_region(state, {'ranked_regions':[proposal]})
        result = {'query':'target','config':{},'usage':usage,'nodes':[], 'trace':[],
            'ranked_clusters':[], 'ranked_regions':[], 'stop_reason':'exhausted',
            'error':'','elapsed_seconds':1., 'final_answer':final}
        with tempfile.TemporaryDirectory() as directory:
            path = save_search_result(result, np.asarray(source), directory)
            saved = json.loads(path.read_text())['final_answer']
            with Image.open(Path(directory)/saved['image_path']) as actual:
                np.testing.assert_array_equal(np.asarray(actual), seen[0])
            self.assertNotIn('input_pixels', saved)
            self.assertEqual(saved['selected_candidate_id'], 'node_7:candidate_2')
        self.assertEqual(usage['answer_generations'],1)


if __name__ == '__main__':
    unittest.main()
