"""Preserve crop-to-scene geometry without target annotations or inferred answers."""
import json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion


def build_region_context(source,bbox,mask=None,*,node_id=None,max_side=768):
    source=source.convert('RGB');width,height=source.size
    x0,y0,x1,y1=map(int,bbox)
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):
        raise ValueError('Region bbox must lie inside the original image')
    if mask is not None and np.asarray(mask).shape!=(y1-y0,x1-x0):
        raise ValueError('Region mask does not match bbox extent')
    nx0,ny0,nx1,ny1=x0/width,y0/height,x1/width,y1/height
    cx,cy=(nx0+nx1)/2,(ny0+ny1)/2
    horizontal='left' if cx<1/3 else 'right' if cx>2/3 else 'center'
    vertical='top' if cy<1/3 else 'bottom' if cy>2/3 else 'middle'
    metadata={'node_id':node_id,'original_image_size':[width,height],
        'region_bbox_xyxy':[x0,y0,x1,y1],
        'region_bbox_normalized_xyxy':[nx0,ny0,nx1,ny1],
        'clean_crop_size':[x1-x0,y1-y0],
        'image_position_of_region_center':vertical+'-'+horizontal,
        'image_coverage_percent':{'x':[100*nx0,100*nx1],'y':[100*ny0,100*ny1]},
        'coordinate_frame':'original image; origin top-left; x increases right; y increases down',
        'crop_to_global':'global_x = region_x0 + crop_x; global_y = region_y0 + crop_y (first undo any display resize)',
        'semantic_caution':'Image position does not by itself establish position relative to an intersection or object. Use scene overview.'}
    overview=source.copy();overview.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    sx,sy=overview.width/width,overview.height/height
    a,b,c,d=round(x0*sx),round(y0*sy),round(x1*sx),round(y1*sy)
    a,b=min(overview.width-1,max(0,a)),min(overview.height-1,max(0,b));c,d=min(overview.width,max(a+1,c)),min(overview.height,max(b+1,d))
    if mask is not None:
        mask=np.asarray(mask,dtype=bool);metadata['active_mask_fraction']=float(mask.mean())
        if not mask.all():
            small=np.asarray(Image.fromarray(mask).resize((c-a,d-b),Image.Resampling.NEAREST),dtype=bool)
            boundary=small & ~binary_erosion(small,iterations=2,border_value=0)
            pixels=np.array(overview);pixels[b:d,a:c][boundary]=[255,210,0];overview=Image.fromarray(pixels)
            metadata['mask_outline']='yellow: active merged-cluster mask, not target annotation'
    draw=ImageDraw.Draw(overview);draw.rectangle((a,b,c-1,d-1),outline=(0,230,255),width=3)
    draw.text((a+4,min(overview.height-14,b+4)),f'REGION N{node_id}' if node_id is not None else 'CURRENT REGION',fill='white',stroke_width=1,stroke_fill='black')
    metadata['overview_size']=list(overview.size)
    metadata['overview_region_bbox_xyxy']=[a,b,c,d]
    return {'overview':overview,'metadata':metadata}


SPATIAL_INSTRUCTIONS="""You receive TWO images of the SAME original scene.
IMAGE 1 is a small FULL-SCENE OVERVIEW. The cyan rectangle locates the current
region in the original image. A yellow contour, if present, is the active merged
cluster mask. These are navigation annotations, not an object/target box and not
object colors. IMAGE 2 is the CLEAN ENLARGED REGION inside that rectangle.
Use IMAGE 1 to establish global position and reference-object relationships.
Use IMAGE 2 to inspect the requested target and its appearance. A target can be
upper-right inside IMAGE 2 while still bottom-left of the intersection in IMAGE 1.
Directions in the question refer to the ORIGINAL scene/reference object, not to
the edges of IMAGE 2. Image coordinates alone do not establish position relative
to an intersection: inspect the overview. Coordinate metadata is geometry only.
The requested target MUST be visible in IMAGE 2. Seeing the target elsewhere in
IMAGE 1 is NOT enough to accept this region or answer from an unrelated region.
Do not infer hidden details or use annotation colors as object attributes.
"""


def spatial_verification_prompt(question,metadata):
    return SPATIAL_INSTRUCTIONS+"""Evaluate whether the original question is answerable using this region's target
detail together with the full-scene spatial context. Judge the target named in
the original question, not merely a reference. SAM3 detections are not required.
Return ONLY JSON with answer exactly Yes, No or Uncertain; score from 0 to 100;
and reason briefly identifying target and evidence. Yes requires visible target,
required identity/relations, and enough detail to answer. No means clear absence
or contradiction; Uncertain means insufficient evidence.
Region mapping: """+json.dumps(metadata)+"\nOriginal question: "+json.dumps(question)


def spatial_answer_prompt(question,metadata):
    return SPATIAL_INSTRUCTIONS+"""Answer the ORIGINAL QUESTION about the requested target in IMAGE 2.
Use the overview to interpret location and relationships; use the clean region
for appearance. If IMAGE 2 lacks the requested target or sufficient evidence,
reply exactly Uncertain. Otherwise give one concise answer, without explanation.
Region mapping: """+json.dumps(metadata)+"\nOriginal question: "+json.dumps(question)
