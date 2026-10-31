"""Pinned SAM3 loading, candidate selection, and mask postprocessing."""

from collections.abc import Callable
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray
import torch

from ..core.contracts import SAM3Detection
from ..core.events import SAM3CandidateRecord
from ..core.imaging import to_pil_rgb


def load_sam3_model(device: str = "auto"):
    """Reuse the project's pinned SAM3 loader, loading weights only on request."""
    scripts_path = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from scripts.vision_backends import load_sam3

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_sam3(device=device, dtype=torch.float32)


def _validate_inputs(
    prompt: str,
    threshold: float,
    mask_threshold: float,
    max_detections: int,
) -> None:
    if not prompt.strip():
        raise ValueError("Provide a short concept prompt, such as 'person'.")
    if not 0 <= threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("Thresholds must be in [0, 1].")
    if type(max_detections) is not int or max_detections < 0:
        raise ValueError("max_detections must be a nonnegative integer.")


def _retained_indices(
    scores: torch.Tensor, threshold: float, max_detections: int
) -> torch.Tensor:
    indices = torch.nonzero(scores[0] > threshold).flatten()
    indices = indices[torch.argsort(scores[0, indices], descending=True)]
    return indices[:max_detections] if max_detections else indices


def _candidate_records(
    outputs,
    scores: torch.Tensor,
    threshold: float,
    retained: set[int],
) -> list[SAM3CandidateRecord]:
    native_boxes = outputs.pred_boxes[0].detach().float().cpu().tolist()
    native_scores = scores[0].detach().float().cpu().tolist()
    records: list[SAM3CandidateRecord] = []
    for query_index, (box, confidence) in enumerate(zip(native_boxes, native_scores)):
        decision = (
            "below_threshold"
            if confidence <= threshold
            else "retained_before_nms"
            if query_index in retained
            else "proposal_limit"
        )
        records.append(
            {
                "query_index": query_index,
                "normalized_bbox": (box[0], box[1], box[2], box[3]),
                "confidence": confidence,
                "retained": query_index in retained,
                "decision": decision,
            }
        )
    return records


def _postprocess_batch(
    outputs,
    processor,
    batch: torch.Tensor,
    threshold: float,
    mask_threshold: float,
    target_size: tuple[int, int],
):
    cpu_outputs = SimpleNamespace(
        pred_logits=outputs.pred_logits[:, batch].detach().float().cpu(),
        pred_boxes=outputs.pred_boxes[:, batch].detach().float().cpu(),
        pred_masks=outputs.pred_masks[:, batch].detach().float().cpu(),
        presence_logits=outputs.presence_logits.detach().float().cpu()
        if outputs.presence_logits is not None
        else None,
    )
    return processor.post_process_instance_segmentation(
        cpu_outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[target_size],
    )[0]


def _detections_from_result(
    result,
    rgb: NDArray[np.uint8],
    prompt: str,
) -> list[SAM3Detection]:
    height, width = rgb.shape[:2]
    detections: list[SAM3Detection] = []
    for box, score, mask in zip(result["boxes"], result["scores"], result["masks"]):
        x1, y1, x2, y2 = box.detach().float().cpu().numpy()
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(score.detach().cpu())
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "score": confidence,
                "sam3_score": confidence,
                "mask": mask.detach().cpu().numpy().astype(bool),
                "image": rgb[y1:y2, x1:x2].copy(),
                "prompt": prompt,
            }
        )
    return detections


@torch.inference_mode()
def exec_sam3_on_image(
    image: NDArray,
    model,
    processor,
    prompt: str = "person",
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    max_detections: int = 0,
    on_candidates: Callable[[list[SAM3CandidateRecord]], None] | None = None,
) -> list[SAM3Detection]:
    """Return detections in input-crop coordinates, with exclusive box maxima.

    Args:
        image: RGB pixels supplied to SAM3, including any existing black mask.
        model: Loaded SAM3 image detector on the desired device.
        processor: SAM3 processor paired with model.
        prompt: Text concept to detect.
        threshold: Minimum detector confidence passed to postprocessing.
        mask_threshold: Threshold for converting predicted masks to binary masks.
        max_detections: Keep the strongest N detections before resizing masks;
            zero retains all detections. Full-resolution mask resizing uses CPU
            batches to avoid allocating a 4K mask stack on the GPU.
        on_candidates: Optional observer for every native box before NMS.
    """
    _validate_inputs(prompt, threshold, mask_threshold, max_detections)

    pil_image = to_pil_rgb(image)
    rgb = np.asarray(pil_image)
    height, width = rgb.shape[:2]
    parameter = next(model.parameters())
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    inputs = {
        key: value.to(
            device=parameter.device,
            dtype=parameter.dtype if value.is_floating_point() else value.dtype,
        )
        for key, value in inputs.items()
    }
    model.eval()
    outputs = model(**inputs)
    scores = outputs.pred_logits.sigmoid()
    if outputs.presence_logits is not None:
        scores = scores * outputs.presence_logits.sigmoid()

    indices = _retained_indices(scores, threshold, max_detections)
    if on_candidates is not None:
        retained = set(indices.detach().cpu().tolist())
        on_candidates(_candidate_records(outputs, scores, threshold, retained))

    detections: list[SAM3Detection] = []
    for batch in indices.split(4):
        result = _postprocess_batch(
            outputs,
            processor,
            batch,
            threshold,
            mask_threshold,
            (height, width),
        )
        detections.extend(_detections_from_result(result, rgb, prompt))
    return sorted(
        detections,
        key=lambda record: record["sam3_score"] or 0.0,
        reverse=True,
    )
