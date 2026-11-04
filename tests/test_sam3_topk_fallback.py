"""SAM3 fallback regression tests: raw-score preservation and downstream survival."""
from types import SimpleNamespace
import numpy as np
import torch
from transformers import Sam3ImageProcessor
from algorithm.models.sam3 import exec_sam3_on_image
from algorithm.search.config import default_search_config
from algorithm.search.evaluation import detect_region


class FakeModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.scores=torch.tensor(scores)
    def forward(self, **kwargs):
        count=len(self.scores)
        return SimpleNamespace(pred_logits=torch.logit(self.scores)[None], presence_logits=None,
            pred_boxes=torch.tensor([.1,.1,.7,.7]).expand(1,count,4),
            pred_masks=torch.ones((1,count,4,4))*10)


class FakeInputProcessor:
    def __init__(self):self.post=Sam3ImageProcessor()
    def __call__(self, **kwargs):return {'pixel_values':torch.zeros((1,3,8,8))}
    def post_process_instance_segmentation(self,*args,**kwargs):
        return self.post.post_process_instance_segmentation(*args,**kwargs)


class TestSAM3TopK:
    def test_no_passing_score_retains_ten_without_promoting_confidence(self):
        records=[];scores=[.001*i for i in range(1,13)]
        found=exec_sam3_on_image(np.zeros((32,32,3),dtype=np.uint8),FakeModel(scores),FakeInputProcessor(),
            prompt='turquoise-cab cement mixer',threshold=.25,max_detections=16,fallback_top_k=10,
            on_candidates=lambda r:records.extend(r))
        assert len(found)==10
        assert abs(found[0]['sam3_score']-.012)<1e-6
        assert all(d['sam3_score']<.25 and d['sam3_selection']=='top_k_fallback' for d in found)
        assert sum(r['decision']=='top_k_fallback' for r in records)==10
        assert all(d['mask'].shape==(32,32) for d in found)

    def test_passing_box_does_not_trigger_fallback(self):
        found=exec_sam3_on_image(np.zeros((32,32,3),dtype=np.uint8),FakeModel([.01,.03,.4]),FakeInputProcessor(),
            threshold=.25,max_detections=16,fallback_top_k=10)
        assert len(found)==1 and found[0]['sam3_selection']=='above_threshold'
        assert abs(found[0]['sam3_score']-.4)<1e-6

    def test_disabled_fallback_preserves_empty_result(self):
        assert exec_sam3_on_image(np.zeros((32,32,3),dtype=np.uint8),FakeModel([.01,.02]),FakeInputProcessor(),threshold=.25)==[]


class TestSearchFallbackFilter:
    def test_retains_only_marked_low_score_fallbacks_and_counts_call(self):
        config=default_search_config();config.update(detector_prompt_mode='references',sam3_threshold=.25,sam3_fallback_top_k=10,
            reference_plan={'target':'car','references':[{'id':'reference_1','prompt':'cement mixer'}]},reference_detections=[])
        def detector(*args):return [{'bbox':(1,1,5,5),'sam3_score':.058,'sam3_selection':'top_k_fallback'},
                                    {'bbox':(6,6,9,9),'sam3_score':.02}]
        state=SimpleNamespace(settings=config,usage={},query='car',detection_cache={},detector=detector,on_event=lambda *a:None,record_cost=lambda *a:None)
        found=detect_region(state,{'node_id':1,'bbox':(10,20,30,40)},np.zeros((20,20,3),dtype=np.uint8))
        assert len(found)==1 and found[0]['sam3_score']==.058
        assert state.usage['sam3_top_k_fallback_calls']==1
        assert state.settings['reference_detections'][0]['sam3_selection']=='top_k_fallback'
