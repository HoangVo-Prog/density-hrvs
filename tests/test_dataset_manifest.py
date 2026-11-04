import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from PIL import Image
from algorithm.core.dataset_manifest import load_inference_questions, load_evaluation_rows
from algorithm.core.data import load_dataset_image


class DatasetManifestTests(unittest.TestCase):
    def test_new_export_projects_only_question_inputs_and_keeps_three_digit_image_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            qa={'question_id':'000_q0001','question':'What color is the bag?',
                'answer':'LABEL_MUST_NOT_REACH_MODEL','target_bbox':{'x':1,'y':2,'width':3,'height':4},
                'target_point_ids':[99]}
            data={'images':[{'image_id':'internal-id','file_name':'000.jpg','qa_annotations':[qa]}]}
            (root/'annotations.json').write_text(json.dumps(data))
            rows=load_inference_questions(root)
            self.assertEqual(rows,[{'image_id':'000','question_id':'000_q0001','question':qa['question']}])
            self.assertNotIn('LABEL_MUST_NOT_REACH_MODEL',json.dumps(rows))
            self.assertNotIn('target_bbox',json.dumps(rows))
            self.assertEqual(load_evaluation_rows(root)[0]['target_bbox'],qa['target_bbox'])
            buffer=io.BytesIO();Image.new('RGB',(9,7),'red').save(buffer,format='PNG')
            with zipfile.ZipFile(root/'images.zip','w') as archive:archive.writestr('images/000.jpg',buffer.getvalue())
            self.assertEqual(load_dataset_image(root,rows[0]['image_id']).size,(9,7))

    def test_legacy_id_padding_and_manifest_remain_compatible(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);Image.new('RGB',(4,5),'blue').save(root/'0063.png')
            (root/'hrvs_bbox_samples_official.json').write_text(json.dumps([
                {'image_id':'0063','question_id':'0063_q01','question':'Color?',
                 'answer':'Blue','target_bbox':{'x':0,'y':0,'width':4,'height':5}}]))
            self.assertEqual(load_dataset_image(root,'63').size,(4,5))
            self.assertEqual(load_inference_questions(root)[0]['image_id'],'0063')
            self.assertEqual(load_evaluation_rows(root)[0]['answer'],'Blue')
