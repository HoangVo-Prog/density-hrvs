"""Optional serialized CPU/GPU staging without changing model weights or inputs."""
from functools import wraps
from threading import RLock
import torch

_lock=RLock()


def staged_callable(function, move, device):
    @wraps(function)
    def call(*args, **kwargs):
        with _lock:
            move(device)
            try:
                return function(*args, **kwargs)
            finally:
                move('cpu')
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
    return call


def staged_scorer(scorer, device):
    move=getattr(scorer,'set_device',None)
    if not callable(move):raise ValueError('Staged scorer requires a movable native model')
    move('cpu')
    torch.cuda.empty_cache()
    wrapped=staged_callable(scorer,move,device)
    for name in ('resolve','rewrite','resolve_structured','answer','answer_evidence','score_candidates','compare_candidates'):
        method=getattr(scorer,name,None)
        if callable(method):setattr(wrapped,name,staged_callable(method,move,device))
    for name in ('model_id','model_revision','quantization'):
        setattr(wrapped,name,getattr(scorer,name))
    return wrapped
