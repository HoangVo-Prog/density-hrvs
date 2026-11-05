"""Next-token binary answerability probabilities, never generated text ratings."""
import torch

SCORE_SOURCE = "yes_no_token_probability"


def yes_no_token_ids(tokenizer):
    ids = {}
    for label in ("Yes", "No"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected one next token for {label!r}, got {encoded}")
        ids[label] = int(encoded[0])
    if ids["Yes"] == ids["No"]:
        raise ValueError("Yes and No must have distinct token IDs")
    return ids


def score_yes_no_logits(logits, token_ids):
    """Return P(Yes | next token in {Yes,No}) plus full-vocabulary evidence.

    Uses the final prompt position BEFORE any answer or rationale is generated.
    Float64 normalization preserves small differences around the decision point.
    The full-vocabulary probability is recorded separately, never conflated with
    the two-token normalization used for the region score.
    """
    values = logits.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("Expected finite next-token vocabulary logits")
    y, n = token_ids["Yes"], token_ids["No"]
    binary = torch.softmax(values[[y, n]], dim=-1)
    full = torch.softmax(values, dim=-1)
    p_yes, p_no = binary.tolist()
    answer = "Yes" if p_yes > p_no else "No" if p_no > p_yes else "Uncertain"
    return {
        "answer": answer, "score": p_yes, "score_source": SCORE_SOURCE,
        "probability_definition": "P(next_token=Yes | next_token in {Yes,No})",
        "yes_probability": p_yes, "no_probability": p_no,
        "yes_probability_full_vocabulary": full[y].item(),
        "no_probability_full_vocabulary": full[n].item(),
        "token_ids": dict(token_ids),
        "token_logits": {"Yes": values[y].item(), "No": values[n].item()},
        "reason": "Answerability score from next-token Yes/No logits; no generated rationale.",
    }
