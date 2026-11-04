"""Topology, provenance and pixel preservation of the new partitioner."""
import numpy as np
from scipy import ndimage
from algorithm.search.adjacent_merge import split_merge_regions


def test_tiny_piece_merges_across_kmeans_label_without_losing_pixels():
    labels=np.zeros((10,20),dtype=int);labels[:,10:]=1;labels[4,4]=2
    active=np.ones_like(labels,dtype=bool);features=np.zeros((*labels.shape,3))
    merged,info=split_merge_regions(labels,features,active,max_regions=2)
    assert info['initial_components']==3 and info['final_regions']==2
    assert merged[4,4]==merged[4,3] and merged[4,4]!=merged[4,15]
    assert np.array_equal(merged>0,active)
    assert sorted(x for members in info['membership'].values() for x in members)==[1,2,3]


def test_distant_same_label_is_split_and_not_joined_across_gap():
    labels=np.zeros((10,30),dtype=int);labels[:,10:20]=1
    merged,info=split_merge_regions(labels,np.zeros((10,30,3)),np.ones_like(labels,bool),
                                   max_regions=3,small_fraction=.001,max_bbox_fraction=.5)
    assert info['initial_components']==3
    assert merged[5,1]!=merged[5,28]
    for value in np.unique(merged):
        assert ndimage.label(merged==value)[1]==1


def test_inactive_hole_and_isolated_small_region_are_preserved():
    labels=np.zeros((10,20),dtype=int);active=np.zeros_like(labels,bool)
    active[:,:8]=True;active[4,18]=True;active[3,3]=False
    merged,_=split_merge_regions(labels,np.zeros((10,20,3)),active)
    assert np.array_equal(merged>0,active)
    assert merged[4,18]!=merged[4,2]


def test_box_limit_prevents_chain_merging_entire_parent():
    labels=np.tile(np.arange(10),(10,1))
    merged,_=split_merge_regions(labels,np.zeros((10,10,3)),np.ones_like(labels,bool),max_regions=2)
    for value in np.unique(merged):
        y,x=np.where(merged==value)
        assert (x.max()-x.min()+1)*(y.max()-y.min()+1)<=65
