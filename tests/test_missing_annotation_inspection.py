"""Missing evaluation boxes must not convert completed inference into a failure."""
import json
from algorithm.reporting import inspection


class TestMissingAnnotationInspection:
    def test_missing_ground_truth_box_preserves_model_answer_and_marks_iou_unavailable(self,tmp_path,monkeypatch):
        run=tmp_path/'run';run.mkdir();report={'query':'question','stop_reason':'local_budget','elapsed_seconds':2.,
            'usage':{'api_calls':0,'reported_cost_usd':0.},'abstention':{'answer':'Uncertain'}}
        (run/'results.json').write_text(json.dumps(report))
        monkeypatch.setattr(inspection,'load_evaluation_rows',lambda root:[{'image_id':'039','question_id':'039_q0001','question':'What color?','answer':'Light pink','target_bbox':None}])
        def forbidden(*args):raise AssertionError('No image/ground-truth drawing needed for missing target annotation')
        monkeypatch.setattr(inspection,'load_dataset_image',forbidden)
        result=inspection.make_inspection(run,tmp_path/'inspection',data_dir=tmp_path,image_id='039',question_id='039_q0001')
        assert result['predicted_answer']=='Uncertain' and result['iou_available'] is False
        assert result['target_bbox'] is None
        assert (tmp_path/'inspection/index.html').exists()
        assert json.loads((run/'results.json').read_text())==report
