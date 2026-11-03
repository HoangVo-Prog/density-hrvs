"""Pinned, lazy Grounding DINO box proposals with truthful backend provenance."""
from functools import lru_cache
from threading import RLock

import numpy as np
import torch
from PIL import Image

GDINO_ID = 'IDEA-Research/grounding-dino-tiny'
GDINO_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
_lock = RLock()


@lru_cache(maxsize=1)
def load_grounding_dino(device='cuda', cache_dir=None):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    if device == 'auto':device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with _lock:
        processor = AutoProcessor.from_pretrained(GDINO_ID, revision=GDINO_REVISION,
            cache_dir=cache_dir, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID,
            revision=GDINO_REVISION, cache_dir=cache_dir, local_files_only=True).to(device).eval()
    return model, processor


@torch.inference_mode()
def detect_grounding_dino(pixels, prompt, *, device='cuda', cache_dir=None,
                          threshold=.25, text_threshold=.25, limit=8, on_candidates=None):
    if not prompt.strip() or not 0 <= threshold <= 1 or not 0 <= text_threshold <= 1 or limit < 1:
        raise ValueError('Invalid Grounding DINO detection settings')
    with _lock:
        model, processor = load_grounding_dino(device, cache_dir)
        image = Image.fromarray(np.asarray(pixels, dtype=np.uint8))
        inputs = processor(images=image, text=prompt.rstrip('. ')+'.', return_tensors='pt').to(model.device)
        outputs = model(**inputs)
        raw = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
            threshold=0., text_threshold=text_threshold, target_sizes=[image.size[::-1]])[0]
        records=[];detections=[]
        order = raw['scores'].argsort(descending=True).tolist()
        for index in order:
            box = raw['boxes'][index].detach().cpu().tolist()
            score = float(raw['scores'][index])
            keep = score > threshold and len(detections) < limit
            records.append({'query_index':index,'bbox':box,'confidence':score,
                            'decision':'retained' if keep else 'below_threshold' if score <= threshold else 'proposal_limit'})
            if not keep:continue
            x1,y1,x2,y2 = box
            bounds = (max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),
                      min(image.width,int(np.ceil(x2))),min(image.height,int(np.ceil(y2))))
            a,b,c,d=bounds
            if a>=c or b>=d:continue
            # This is explicitly a rectangular box mask, not a segmentation prediction.
            mask=np.zeros(pixels.shape[:2],dtype=bool);mask[b:d,a:c]=True
            detections.append({'bbox':bounds,'score':score,'sam3_score':None,
                'detector_score':score,'proposal_backend':GDINO_ID,'model_revision':GDINO_REVISION,
                'mask_kind':'box_only','mask':mask,'image':pixels[b:d,a:c].copy(),'prompt':prompt})
        if on_candidates is not None:on_candidates(records)
        return detections
