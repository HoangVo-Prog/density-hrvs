"""Paired partition-only K sweep; one full-image DINO extraction per image.

No VLM answers, target boxes or answer labels are used. Silhouette measures
feature separation, not target localization or question-answer accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
from sklearn.metrics import silhouette_score

from ..core.data import load_dataset_image
from ..core.dataset_manifest import load_inference_questions
from ..models.regions import (DINOV3_ID,DINOV3_REVISION,load_dinov3,
    exec_dinov3_pca_on_image,optimize_slic_n_segments,cluster_slic_regions)
from ..search.adjacent_merge import split_merge_regions
from ..search.config import default_search_config as default_config
from ..search.partitioning import _split_quality,_tight_child,spatial_children


def geometry(labels):
    node={'bbox':(0,0,labels.shape[1],labels.shape[0]),'mask':np.ones(labels.shape,bool)}
    children=[_tight_child(labels==v,node) for v in np.unique(labels)]
    _,reasons,qualities=_split_quality(children,node,default_config())
    return node,children,reasons,qualities


def score(features,labels,sample):
    y=labels.ravel()[sample];n=len(np.unique(y))
    if n<2:return {'value':None,'reason':'Only one sampled label; silhouette requires at least two.'}
    if n>=len(y):return {'value':None,'reason':'One label per sampled point.'}
    return {'value':float(silhouette_score(features[sample],y)), 'reason':None}


def run(args):
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    config=default_config();model=processor=None;results=[]
    image_ids={q['image_id'] for q in load_inference_questions(args.data_dir)}
    extensions={'.jpg','.jpeg','.png','.webp'}
    image_ids.update(p.stem for p in args.data_dir.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    for archive in args.data_dir.glob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            image_ids.update(Path(n).stem for n in z.namelist() if Path(n).suffix.lower() in extensions)
    ids=sorted(image_ids)
    specification={'data_dir':str(args.data_dir.resolve()),'image_ids':ids,'cluster_counts':[1,3,5,7],
        'dino_model':DINOV3_ID,'dino_revision':DINOV3_REVISION,'max_side':1024,
        'spatial_weight':.75,'silhouette_sample':args.sample_size,'random_seed':42,
        'feature_space':'three normalized DINO PCA colors plus 0.75-scaled normalized x,y',
        'note':'Partition-only experiment. No answer/localization improvement is inferred from silhouette.'}
    manifest=out/'specification.json'
    if manifest.exists() and json.loads(manifest.read_text())!=specification:
        raise ValueError('Experiment specification differs; choose a new output directory')
    manifest.write_text(json.dumps(specification,indent=2)+'\n')
    if args.limit:ids=ids[:args.limit]
    for image_id in ids:
        folder=out/image_id;folder.mkdir(exist_ok=True);completed=folder/'comparison.json'
        if completed.exists():
            results.append(json.loads(completed.read_text()));continue
        started=time.perf_counter();image=load_dataset_image(args.data_dir,image_id)
        digest=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        cache=folder/'dino_pca.npz';calls=0
        if cache.exists():
            saved=np.load(cache)
            if str(saved['image_sha256'])!=digest:raise ValueError('Image changed beneath feature cache')
            fmap=saved['features']
        else:
            if model is None:model,processor=load_dinov3(device='cuda')
            raw=exec_dinov3_pca_on_image(np.asarray(image),model,processor,max_side=1024);calls=1
            h,w=raw.shape[:2];scale=min(1.,1024/max(h,w))
            # Nearest resampling retains the same PCA representation for both arms.
            ys=np.minimum((np.arange(max(1,round(h*scale)))/scale).astype(int),h-1)
            xs=np.minimum((np.arange(max(1,round(w*scale)))/scale).astype(int),w-1)
            fmap=raw[ys[:,None],xs[None,:]].copy();del raw
            temp=folder/'dino_pca.partial.npz'
            np.savez_compressed(temp,features=fmap,image_sha256=digest);temp.replace(cache)
            (folder/'feature_provenance.json').write_text(json.dumps({'image_sha256':digest,
                'image_size':list(image.size),'dino_model':DINOV3_ID,'revision':DINOV3_REVISION,
                'physical_encoder_calls':1,'full_image_once':True})+'\n')
        segmentation=optimize_slic_n_segments(fmap,config['slic'],config['n_segments_values'],config['entropy_bins'])
        h,w=fmap.shape[:2];yy,xx=np.indices((h,w));vectors=np.column_stack((fmap.reshape(-1,3),
            .75*xx.ravel()/max(1,w-1),.75*yy.ravel()/max(1,h-1)))
        rng=np.random.default_rng(42);sample=rng.choice(h*w,min(args.sample_size,h*w),replace=False)
        comparisons=[];maps={}
        for k in (1,3,5,7):
            if k==1:labels=np.zeros((h,w),dtype=np.int32)
            else:labels=cluster_slic_regions(fmap,segmentation['labels'],n_clusters=k,spatial_weight=.75)['labels']
            merged,info=split_merge_regions(labels,fmap,np.ones((h,w),bool),max_regions=max(2,k))
            for name,partition in [('current',labels),('adjacent_merge',merged)]:
                node,children,reasons,qualities=geometry(partition)
                fallback=any(reasons.values()) or len(children)>max(2,k)
                # K=1 is an explicit unsplit baseline, not an implicit four-tile arm.
                if k==1:fallback=False
                effective=partition.copy()
                if fallback:
                    effective=np.zeros((h,w),np.int32)
                    for i,child in enumerate(spatial_children(node),1):
                        a,b,c,d=child['bbox'];effective[b:d,a:c][child['mask']]=i
                maps[f'{name}_k{k}_raw']=partition;maps[f'{name}_k{k}_effective']=effective
                comparisons.append({'method':name,'k':k,'raw_region_count':len(children),
                    'effective_region_count':len(np.unique(effective)),'spatial_fallback':fallback,
                    'fallback_reasons':reasons,'quality':qualities,
                    'raw_silhouette':score(vectors,partition,sample),
                    'effective_silhouette':score(vectors,effective,sample),
                    'merge_info':info if name=='adjacent_merge' else None})
        np.savez_compressed(folder/'partition_maps.npz',**maps)
        record={'image_id':image_id,'image_sha256':digest,'feature_shape':list(fmap.shape),
                'dino_calls_this_invocation':calls,'elapsed_seconds':time.perf_counter()-started,
                'comparisons':comparisons}
        temp=folder/'comparison.partial.json';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(completed)
        results.append(record)
        summary={'expected_images':len(specification['image_ids']),'completed_images':len(results),
                 'paired_configurations_per_image':8,'physical_dino_calls_saved':len(list(out.glob('*/feature_provenance.json'))),
                 'status':'running','aggregates':[]}
        for name in ('current','adjacent_merge'):
            for k in (1,3,5,7):
                rows=[c for r in results for c in r['comparisons'] if c['method']==name and c['k']==k]
                valid=[c['effective_silhouette']['value'] for c in rows if c['effective_silhouette']['value'] is not None]
                summary['aggregates'].append({'method':name,'k':k,'images':len(rows),
                    'mean_effective_silhouette':float(np.mean(valid)) if valid else None,
                    'spatial_fallback_images':sum(c['spatial_fallback'] for c in rows),
                    'mean_effective_regions':float(np.mean([c['effective_region_count'] for c in rows]))})
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f"IMAGE {len(results)}/{len(ids)} {image_id}: paired K=1,3,5,7 done; new DINO calls={calls}",flush=True)
    if (out/'summary.json').exists():
        summary=json.loads((out/'summary.json').read_text());summary['status']='finished' if len(results)==len(specification['image_ids']) else 'partial_limit'
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--sample-size',type=int,default=512)
    run(p.parse_args())


if __name__=='__main__':main()
