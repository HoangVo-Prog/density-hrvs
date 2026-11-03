import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from algorithm.tools.experiment_baseline import selection_inputs, verify_snapshot


class BaselineIntegrityTests(unittest.TestCase):
    def test_changed_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / 'result.json'
            evidence.write_text('{"answer":"blue"}')
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps({
                'files': {'result.json': digest}, 'metrics': {'questions': 1}}))
            self.assertEqual(verify_snapshot(root), {'questions': 1})
            evidence.write_text('{"answer":"red"}')
            with self.assertRaisesRegex(ValueError, 'Frozen baseline changed'):
                verify_snapshot(root)

    def test_diagnostic_inference_inputs_exclude_evaluation_labels(self):
        rows = [{'question_id': 'a', 'image_id': 'b', 'question': 'What color?',
                 'answer': 'blue', 'target_bbox': [1, 2, 3, 4], 'joint_correct': True}]
        self.assertEqual(selection_inputs(rows), [
            {'question_id': 'a', 'image_id': 'b', 'question': 'What color?'}])


if __name__ == '__main__':
    unittest.main()
