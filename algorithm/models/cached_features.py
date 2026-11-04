"""Reuse one pinned full-image DINO PCA extraction for all search crops."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .regions import DINOV3_ID,DINOV3_REVISION


class CachedFullImageFeatures:
    def __init__(self,image,folder,encode):
        self.image=np.asarray(image);self.folder=Path(folder);self.encode=encode
        self.features=None;self.encoder_calls=0;self.cache_reads=0

    def __call__(self,pixels):
        raise RuntimeError('Cached features require explicit global region coordinates')

    def _load(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/'dino_pca.npz'
        digest=hashlib.sha256(self.image.tobytes()).hexdigest()
        if path.exists():
            metadata=json.loads((self.folder/'feature_provenance.json').read_text())
            if metadata.get('revision')!=DINOV3_REVISION or metadata.get('dino_model')!=DINOV3_ID:
                raise ValueError('Cached DINO checkpoint differs')
            with np.load(path,allow_pickle=False) as cache:
                if str(cache['image_sha256'])!=digest:raise ValueError('Cached image content differs')
                self.features=cache['features']
            self.cache_reads+=1
        else:
            raw=self.encode(self.image);self.encoder_calls+=1
            h,w=raw.shape[:2];scale=min(1.,1024/max(h,w))
            ys=np.minimum((np.arange(max(1,round(h*scale)))/scale).astype(int),h-1)
            xs=np.minimum((np.arange(max(1,round(w*scale)))/scale).astype(int),w-1)
            self.features=raw[ys[:,None],xs[None,:]].copy()
            tmp=self.folder/'dino_pca.partial.npz'
            np.savez_compressed(tmp,features=self.features,image_sha256=digest);tmp.replace(path)
            (self.folder/'feature_provenance.json').write_text(json.dumps({'image_sha256':digest,
                'image_size':[w,h],'dino_model':DINOV3_ID,'revision':DINOV3_REVISION,
                'physical_encoder_calls':1,'full_image_once':True})+'\n')
        if self.features.ndim!=3 or self.features.shape[2]!=3 or not np.isfinite(self.features).all():
            raise ValueError('Malformed cached feature map')

    def for_region(self,node):
        if self.features is None:self._load()
        x1,y1,x2,y2=node['bbox'];H,W=self.image.shape[:2];h,w=self.features.shape[:2]
        if not 0<=x1<x2<=W or not 0<=y1<y2<=H:raise ValueError('Region outside source image')
        xs=np.minimum(np.arange(x1,x2)*w//W,w-1);ys=np.minimum(np.arange(y1,y2)*h//H,h-1)
        result=self.features[ys[:,None],xs[None,:]].copy()
        result[~node['mask']]=0
        return result
