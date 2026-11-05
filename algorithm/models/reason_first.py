"""Parse compact reason-first model responses without treating text as confidence."""
from .scoring import _json_payload


def parse_reason_first(text, expected_answer=None):
    data = _json_payload(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a reason/answer JSON object")
    reason, answer = data.get("reason"), data.get("answer")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise ValueError("Missing brief reason")
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 300:
        raise ValueError("Missing concise answer")
    if expected_answer is not None and answer != expected_answer:
        raise ValueError("Explanation changed the probability-based verifier decision")
    return {"reason": reason.strip(), "answer": answer.strip()}
