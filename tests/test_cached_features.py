import tempfile
from pathlib import Path
import numpy as np
import pytest
from algorithm.models.cached_features import CachedFullImageFeatures


def test_global_crop_mapping_and_second_question_reuses_cache():
    image=np.zeros((8,10,3),np.uint8)
    raw=np.arange(8*10*3,dtype=np.float32).reshape(8,10,3)/240
    calls=[]
    def encode(pixels):calls.append(1);return raw
    with tempfile.TemporaryDirectory() as d:
        provider=CachedFullImageFeatures(image,d,encode)
        node={'bbox':(3,2,8,6),'mask':np.ones((4,5),bool)}
        node['mask'][1,1]=False
        expected=raw[2:6,3:8].copy();expected[1,1]=0
        np.testing.assert_array_equal(provider.for_region(node),expected)
        np.testing.assert_array_equal(provider.for_region(node),expected)
        second=CachedFullImageFeatures(image,d,lambda _:pytest.fail('Re-encoded same image'))
        np.testing.assert_array_equal(second.for_region(node),expected)
        assert len(calls)==1 and provider.encoder_calls==1 and second.encoder_calls==0
        with pytest.raises(ValueError):second.for_region({'bbox':(-1,0,5,4),'mask':np.ones((4,6),bool)})
        changed=CachedFullImageFeatures(image+1,d,encode)
        with pytest.raises(ValueError,match='content differs'):changed.for_region(node)
