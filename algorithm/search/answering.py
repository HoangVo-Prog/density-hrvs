"""Generate a text answer from the original question and the top-ranked region."""

import json

from ..core.contracts import BoundingBox, FinalAnswer, SearchResult
from ..models.qwen_scoring import (
    ANSWER_PROMPT,
    EVIDENCE_ANSWER_PROMPT,
    QWEN_MODEL,
    QWEN_REVISION,
    clean_answer,
)
from .runtime import RunState
import numpy as np
from PIL import Image
from ..models.evidence import build_answer_evidence, evidence_info
from .eligibility import eligible_regions
from ..core.calls import invoke_counted


def padded_bbox(
    bbox: BoundingBox,
    image_size: tuple[int, int],
    padding: int,
) -> BoundingBox:
    """Expand a box by ``padding`` pixels, clamped to the image."""
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(round(x1)) - padding),
        max(0, int(round(y1)) - padding),
        min(width, int(round(x2)) + padding),
        min(height, int(round(y2)) + padding),
    )


def answer_top_region(state: RunState, result: SearchResult) -> FinalAnswer | None:
    """Answer the original question from the highest-ranked proposal crop.

    The answerer is a single post-search Qwen generation. It is recorded in the
    local-generation ledger but is not subject to the search budget, because the
    search has already stopped and dropping the answer would defeat the stage.
    """
    if not state.settings.get("answer_question", True):
        return None
    if not result["ranked_regions"]:
        return None
    answerer = getattr(state.local_scorer, "answer", None)
    if not callable(answerer):
        return None

    eligible = eligible_regions(result["ranked_regions"], state.settings)
    if not eligible:
        result["abstention"] = {"answer": "Uncertain", "reason": "no_verified_candidate"}
        return None
    best = eligible[0]
    padding = state.settings.get("answer_context_padding", 384)
    crop_bbox = padded_bbox(best["bbox"], state.source_image.size, padding)
    x1, y1, x2, y2 = crop_bbox
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    view_type = state.settings.get("answer_view", "legacy_crop")
    evidence = None
    if view_type == "marked_detail":
        evidence_answerer = getattr(state.local_scorer, "answer_evidence", None)
        if not callable(evidence_answerer):
            raise ValueError("marked_detail requires an answer_evidence callable")
        answerer = evidence_answerer
        evidence = build_answer_evidence(state.source_image, best["bbox"],
            max_side=state.settings.get("verification_max_side", 1536),
            context_padding=padding, detail_padding=state.settings.get("detail_padding", 16))
        crop = Image.fromarray(evidence["image"])
        crop_bbox = evidence["context_bbox"]
        template = EVIDENCE_ANSWER_PROMPT
    else:
        crop = state.source_image.crop(crop_bbox)
        template = ANSWER_PROMPT
    prompt = template.format(question=json.dumps(state.question))
    raw = invoke_counted(answerer, crop, state.question, on_request=state.charge_answer)
    if not isinstance(raw, str):
        raise ValueError("answerer returned a non-string")
    answer = clean_answer(raw)
    return {
        "view_type": view_type,
        "selected_candidate_id": f"node_{best.get('cluster_id', 'unknown')}:candidate_{best.get('candidate_rank', 'unknown')}",
        "evidence_info": evidence_info(evidence) if evidence is not None else {},
        "input_pixels": np.asarray(crop),
        "question": state.question,
        "answer": answer,
        "bbox": best["bbox"],
        "crop_bbox": crop_bbox,
        "proposal_score": best["score"],
        "model_id": QWEN_MODEL,
        "raw_response": raw,
        "prompt": prompt,
        "input_size": crop.size,
        "model_revision": QWEN_REVISION,
    }
