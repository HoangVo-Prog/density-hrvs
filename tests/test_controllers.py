import unittest
import numpy as np
from algorithm.search.tree import _select_path,new_node
from algorithm.search.partitioning import spatial_children
from algorithm.search.pipeline import split_node
from algorithm.search.runtime import RunState
from algorithm.search.config import default_search_config,validate_search_config
from PIL import Image


class ControllerTests(unittest.TestCase):
    def test_bfs_selects_global_shallowest_frontier_instead_of_following_best_branch(self):
        region={'bbox':(0,0,100,100),'mask':np.ones((100,100),dtype=bool)}
        nodes=[new_node(region,0,-1,0),new_node(region,1,0,1),new_node(region,2,0,1),new_node(region,3,1,2)]
        nodes[0]['children']=[1,2];nodes[1]['children']=[3]
        config=default_search_config();config['strategy']='bfs';events=[]
        path,terms=_select_path(nodes,config,lambda n:1.,lambda *a:events.append(a))
        self.assertEqual(path,[0,2]);self.assertEqual(terms['exploration_bonus'],0.)
        nodes[2]['exhausted']=True
        self.assertEqual(_select_path(nodes,config,lambda n:1.,lambda *a:None)[0],[0,1,3])
        self.assertIn('Breadth-first',events[0][1])

    def test_spatial_partition_overlaps_and_never_invokes_feature_encoder(self):
        config=default_search_config();config.update(partition_mode='spatial',strategy='bfs',spatial_overlap=.1)
        validate_search_config(config)
        source=Image.new('RGB',(200,200))
        def forbidden(p):raise AssertionError('DINO was invoked in spatial mode')
        state=RunState(query='person',question='Color?',settings=config,usage={},
            source_image=source,rgb=np.asarray(source),artifacts=None,on_event=lambda *a:None,
            progress=lambda *a:None,custom_scorer=None,local_scorer=None,
            feature_extractor=forbidden,detector=lambda *a:[])
        node=new_node({'bbox':(0,0,200,200),'mask':np.ones((200,200),dtype=bool)},0,-1,0)
        children=split_node(state,node)
        self.assertEqual(len(children),4)
        self.assertGreater(children[0]['bbox'][2],children[1]['bbox'][0])
        self.assertEqual(state.usage.get('feature_encoder_calls',0),0)
        self.assertEqual(children[0]['mask'].shape,(120,120))
        self.assertEqual(spatial_children(node)[0]['bbox'],(0,0,100,100))
