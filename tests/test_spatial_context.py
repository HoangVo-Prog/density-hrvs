"""Crop-to-scene mapping, clean detail routing, spatial cache identity and answer tests."""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from algorithm.models.spatial_context import build_region_context,spatial_verification_prompt
from algorithm.search.answering import answer_top_region
from algorithm.search.pipeline import evaluate_node
from test_question_region import state_with,node,QUESTION


class TestRegionMapping:
    def test_bottom_left_quarter_has_exact_global_and_normalized_coordinates(self):
        source=Image.new('RGB',(3840,2160),(40,50,60));before=source.tobytes()
        result=build_region_context(source,(0,1080,1920,2160),node_id=3)
        m=result['metadata']
        assert m['region_bbox_normalized_xyxy']==[0.,.5,.5,1.]
        assert m['region_bbox_xyxy']==[0,1080,1920,2160]
        assert m['image_position_of_region_center']=='bottom-left'
        assert m['overview_region_bbox_xyxy']==[0,216,384,432]
        assert result['overview'].size==(768,432) and source.tobytes()==before
        assert 'intersection' in m['semantic_caution']

    def test_mask_outline_is_not_a_target_annotation_and_detail_is_not_modified(self):
        source=Image.new('RGB',(100,100),'black');mask=np.zeros((40,40),bool);mask[10:30,10:30]=True
        result=build_region_context(source,(10,10,50,50),mask,node_id=4)
        assert result['metadata']['active_mask_fraction']==.25
        assert 'not target annotation' in result['metadata']['mask_outline']
        assert source.getpixel((20,20))==(0,0,0)
        with pytest.raises(ValueError,match='mask'):
            build_region_context(source,(10,10,50,50),np.ones((2,2),bool))

    def test_one_pixel_region_at_bottom_right_stays_inside_overview(self):
        result=build_region_context(Image.new('RGB',(3840,2160)),(3839,2159,3840,2160),np.ones((1,1),bool))
        a,b,c,d=result['metadata']['overview_region_bbox_xyxy']
        assert 0<=a<c<=768 and 0<=b<d<=432


class TestSpatialRouting:
    def test_question_and_clean_region_plus_correct_overview_are_sent_without_object_queries(self):
        seen=[]
        def verifier(im,q,*,spatial_context,on_request):
            assert q==QUESTION and im.size==(20,20)
            assert np.all(np.asarray(im)==[42,127,210])
            assert spatial_context['metadata']['region_bbox_xyxy']==[10,10,30,30]
            assert 'white structure' not in str(spatial_context['metadata'])
            on_request();seen.append(spatial_context)
            return {'answer':'Yes','score':.9,'reason':'Visible','model_id':'mock'}
        s=state_with(lambda *a:[],verifier);s.settings['region_spatial_context']=True
        e=evaluate_node(s,node());assert len(seen)==1 and e['raw_reward']==.9
        assert 'verification_overview' in s.observed_clusters[0]

    def test_identical_crops_at_different_scene_positions_do_not_share_score_cache(self):
        calls=[]
        def verifier(im,q,*,spatial_context,on_request):
            on_request();calls.append(spatial_context['metadata']['region_bbox_xyxy'])
            return {'answer':'Uncertain','score':.3,'reason':'Context','model_id':'mock'}
        s=state_with(lambda *a:[],verifier);s.settings['region_spatial_context']=True
        s.rgb[:]=42;s.source_image=Image.fromarray(s.rgb)
        n1=node();n2={**node(),'node_id':4,'bbox':(30,10,50,30)}
        evaluate_node(s,n1);evaluate_node(s,n2)
        assert len(calls)==2 and len(s.score_cache)==2

    def test_final_answer_gets_same_scene_mapping_and_selected_clean_region(self):
        def verifier(im,q,*,spatial_context,on_request):
            on_request();return {'answer':'Yes','score':.9,'reason':'Visible','model_id':'mock'}
        s=state_with(lambda *a:[],verifier);s.settings.update(region_spatial_context=True,answer_view='spatial_context')
        evaluate_node(s,node())
        def answer(im,q,*,spatial_context,on_request):
            assert q==QUESTION and im.size==(20,20)
            assert spatial_context['metadata']['region_bbox_xyxy']==list(node()['bbox'])
            on_request();return 'crescent-shaped'
        s.local_scorer.answer_spatial=answer
        result=answer_top_region(s,{'ranked_regions':s.observed_regions})
        assert result['view_type']=='spatial_context' and result['crop_bbox']==node()['bbox']
        assert result['input_overview_pixels'].shape[:2]==(40,50)
        assert 'target MUST be visible in IMAGE 2' in result['prompt']


class TestSpatialArtifactExport:
    def test_saves_both_actual_model_images_without_arrays_in_json(self,tmp_path):
        from algorithm.reporting.artifacts import save_search_result
        import json
        def verifier(im,q,*,spatial_context,on_request):
            on_request();return {'answer':'Yes','score':.9,'reason':'Visible','model_id':'mock',
                'spatial_metadata':spatial_context['metadata']}
        s=state_with(lambda *a:[],verifier);s.settings.update(region_spatial_context=True,answer_view='spatial_context')
        n={**node(),'parent_id':0,'children':[],'visits':1,'reward_sum':.9,'evaluated':True,'exhausted':False}
        evaluate_node(s,n)
        def answer(im,q,*,spatial_context,on_request):on_request();return 'crescent-shaped'
        s.local_scorer.answer_spatial=answer
        result={'ranked_regions':s.observed_regions,'ranked_clusters':s.observed_clusters,
            'nodes':[n],'query':QUESTION,'config':s.settings,'usage':s.usage,'stop_reason':'test',
            'error':'','elapsed_seconds':1.,'trace':[]}
        result['final_answer']=answer_top_region(s,result)
        path=save_search_result(result,s.rgb,tmp_path)
        report=json.loads(path.read_text())
        assert (tmp_path/report['clusters'][0]['overview_input_image_path']).is_file()
        assert (tmp_path/report['final_answer']['overview_image_path']).is_file()
        assert report['final_answer']['spatial_metadata']['region_bbox_xyxy']==[10,10,30,30]
        assert 'input_overview_pixels' not in report['final_answer']
