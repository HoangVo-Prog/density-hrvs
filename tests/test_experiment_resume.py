from pathlib import Path
import tempfile
import unittest
from algorithm.tools.run_experiment import prepare_experiment


class ExperimentResumeTests(unittest.TestCase):
    def test_resume_uses_original_source_and_rejects_tampered_inputs_or_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';data=root/'data';output=root/'output'
            (source/'algorithm').mkdir(parents=True);data.mkdir()
            (source/'algorithm/__init__.py').write_text('# original source')
            (data/'annotations.json').write_text('{}')
            first=prepare_experiment(output,data,{'budget':8},[],source)
            (source/'algorithm/__init__.py').write_text('# changed working branch')
            resumed=prepare_experiment(output,data,{'budget':8},[],source)
            self.assertEqual(first['source_hashes'],resumed['source_hashes'])
            self.assertEqual((output/'frozen_source/algorithm/__init__.py').read_text(),'# original source')
            with self.assertRaises(ValueError):prepare_experiment(output,data,{'budget':9},[],source)
            (output/'frozen_source/algorithm/__init__.py').write_text('# changed snapshot')
            with self.assertRaises(ValueError):prepare_experiment(output,data,{'budget':8},[],source)
