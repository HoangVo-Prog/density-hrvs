"""Official pinned LLMDet proposals with explicit box-only provenance."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from .residency import staged_callable

LLMDET_ID='iSEE-Laboratory/llmdet_tiny'
LLMDET_REVISION='d05199165a19320a9236396e20fca0a5e065189c'


class LLMDetDetector:
    def __init__(self,*,device='cuda',cache_dir=None,threshold=.25,limit=16,staged=True):
        self.processor=AutoProcessor.from_pretrained(LLMDET_ID,revision=LLMDET_REVISION,cache_dir=cache_dir,local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(LLMDET_ID,revision=LLMDET_REVISION,cache_dir=cache_dir,local_files_only=True).eval()
        self.threshold=threshold;self.limit=limit
        self.call=staged_callable(self._detect,self.model.to,device) if staged else self._detect
        if not staged:self.model.to(device)

    def __call__(self,pixels,prompt):
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('LLMDet requires a nonempty locating statement')
        return self.call(pixels,prompt)

    @torch.inference_mode()
    def _detect(self,pixels,prompt):
        image=Image.fromarray(np.array(pixels,dtype=np.uint8,copy=True))
        inputs=self.processor(images=image,text=prompt,return_tensors='pt').to(self.model.device)
        outputs=self.model(**inputs)
        result=self.processor.post_process_grounded_object_detection(outputs,input_ids=inputs.input_ids,threshold=self.threshold,
            target_sizes=[(image.height,image.width)])[0]
        detections=[]
        for index in result['scores'].argsort(descending=True).tolist():
            if len(detections)>=self.limit:break
            box=result['boxes'][index].detach().cpu().tolist()
            x1,y1,x2,y2=box
            a,b,c,d=max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),min(image.width,int(np.ceil(x2))),min(image.height,int(np.ceil(y2)))
            if a>=c or b>=d:continue
            confidence=float(result['scores'][index]);mask=np.zeros(pixels.shape[:2],bool);mask[b:d,a:c]=True
            detections.append({'bbox':(a,b,c,d),'score':confidence,'sam3_score':None,
                'detector_score':confidence,'proposal_backend':LLMDET_ID,'model_revision':LLMDET_REVISION,
                'mask_kind':'box_only','mask':mask,'image':pixels[b:d,a:c].copy(),'prompt':prompt})
        return detections
