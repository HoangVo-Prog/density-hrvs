"""Generate a text answer from the original question and the top-ranked region."""

import json

from ..core.contracts import BoundingBox, FinalAnswer, SearchResult
from ..models.qwen_scoring import (
    ANSWER_PROMPT,
    QWEN_MODEL,
    QWEN_REVISION,
    clean_answer,
)
from .runtime import RunState


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

    best = result["ranked_regions"][0]
    padding = state.settings.get("answer_context_padding", 384)
    crop_bbox = padded_bbox(best["bbox"], state.source_image.size, padding)
    x1, y1, x2, y2 = crop_bbox
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = state.source_image.crop(crop_bbox)
    prompt = ANSWER_PROMPT.format(question=json.dumps(state.question))
    state.usage["local_score_calls"] = state.usage.get("local_score_calls", 0) + 1
    raw = answerer(crop, state.question)
    if not isinstance(raw, str):
        raise ValueError("answerer returned a non-string")
    answer = clean_answer(raw)
    return {
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
