"""Connected-component splitting and adjacency-only search-region merging.

K-means labels are appearance hints, not an ownership constraint. All active
pixels survive. The caller bounds branching using spatial fallback if necessary.
"""
import heapq
import numpy as np
from scipy import ndimage


def split_merge_regions(labels, features, active, *, small_fraction=.02,
                        max_regions=6, max_bbox_fraction=.65, min_occupancy=.25):
    """Return connected region labels and an auditable original-piece map.

    Four-connected pieces may merge across appearance labels, but only across
    a shared edge. Scores favor common boundary, feature similarity and compact
    union boxes. Region count is a soft target: geometry constraints win.
    """
    labels=np.asarray(labels);features=np.asarray(features);active=np.asarray(active,dtype=bool)
    if labels.shape!=active.shape or features.shape[:2]!=labels.shape:
        raise ValueError('Labels, features and active mask must share dimensions')
    if not 0<=small_fraction<=1 or max_regions<2 or not 0<max_bbox_fraction<1 or not 0<min_occupancy<=1:
        raise ValueError('Invalid split-merge thresholds')
    pieces=np.zeros(labels.shape,dtype=np.int32);offset=0
    for value in np.unique(labels[active]):
        local,count=ndimage.label((labels==value)&active)
        occupied=local>0;pieces[occupied]=local[occupied]+offset;offset+=count
    count=offset
    if not count:return pieces,{'initial_components':0,'final_regions':0,'merges':[],'membership':{}}
    area=np.bincount(pieces.ravel(),minlength=count+1)
    objects=ndimage.find_objects(pieces)
    sums=np.stack([np.bincount(pieces.ravel(),weights=features[...,c].ravel(),minlength=count+1)
                   for c in range(features.shape[-1])],axis=1)
    regions={i:{'area':int(area[i]),'sum':sums[i],
                'box':(s[1].start,s[0].start,s[1].stop,s[0].stop),'members':[i], 'version':0}
             for i,s in enumerate(objects,1) if s is not None}
    adjacency={i:{} for i in regions}
    for left,right in [(pieces[:,:-1],pieces[:,1:]),(pieces[:-1,:],pieces[1:,:])]:
        keep=(left!=right)&(left>0)&(right>0)
        pairs=np.sort(np.stack([left[keep],right[keep]],axis=1),axis=1)
        unique,counts=np.unique(pairs,axis=0,return_counts=True)
        for (a,b),boundary in zip(unique,counts):
            a,b=int(a),int(b);n=int(boundary)
            adjacency[a][b]=adjacency[a].get(b,0)+n;adjacency[b][a]=adjacency[b].get(a,0)+n
    parent_area=active.shape[0]*active.shape[1];total=int(active.sum());heap=[];merges=[]

    def offer(a,b):
        if a>b:a,b=b,a
        A,B=regions[a],regions[b]
        x1=min(A['box'][0],B['box'][0]);y1=min(A['box'][1],B['box'][1])
        x2=max(A['box'][2],B['box'][2]);y2=max(A['box'][3],B['box'][3])
        box_area=(x2-x1)*(y2-y1);occupancy=(A['area']+B['area'])/box_area
        if box_area>max_bbox_fraction*parent_area or occupancy<min_occupancy:return
        distance=float(np.linalg.norm(A['sum']/A['area']-B['sum']/B['area']))
        boundary=adjacency[a][b]/max(1,min(A['area'],B['area'])**.5)
        cost=distance+(1-occupancy)+box_area/parent_area-boundary
        small=min(A['area'],B['area'])<=small_fraction*total
        heapq.heappush(heap,(not small,cost,a,b,A['version'],B['version']))

    for a in adjacency:
        for b in adjacency[a]:
            if a<b:offer(a,b)
    while heap:
        _,cost,a,b,va,vb=heapq.heappop(heap)
        if a not in regions or b not in regions:continue
        A,B=regions[a],regions[b]
        if (va,vb)!=(A['version'],B['version']):continue
        small=min(A['area'],B['area'])<=small_fraction*total
        if not small and len(regions)<=max_regions:continue
        box=tuple(f(A['box'][i],B['box'][i]) for i,f in enumerate([min,min,max,max]))
        merges.append({'into':a,'absorbed':b,'small_piece_merge':small,'cost':float(cost)})
        A.update(area=A['area']+B['area'],sum=A['sum']+B['sum'],box=box,
                 members=A['members']+B['members'],version=A['version']+1)
        for neighbor,boundary in list(adjacency[b].items()):
            adjacency[neighbor].pop(b,None)
            if neighbor!=a:
                adjacency[a][neighbor]=adjacency[a].get(neighbor,0)+boundary
                adjacency[neighbor][a]=adjacency[a][neighbor]
        adjacency[a].pop(b,None);del adjacency[b];del regions[b]
        for neighbor in adjacency[a]:offer(a,neighbor)
    mapping=np.zeros(count+1,dtype=np.int32);membership={}
    for new,(old,region) in enumerate(sorted(regions.items()),1):
        mapping[region['members']]=new;membership[str(new)]=region['members']
    return mapping[pieces],{'initial_components':count,'final_regions':len(regions),
                           'merges':merges,'membership':membership}
