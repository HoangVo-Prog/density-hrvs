"""Question-only target derivation shared by the runner and benchmarks."""

import re


def question_target(question: str) -> tuple[str, str]:
    """Derive a target and SAM concept solely from question text."""
    target = re.sub(
        r"^What (?:colors?|pattern|shape) (?:is|are)(?: on)? ",
        "",
        question,
        flags=re.I,
    ).rstrip("?")
    # SAM accepts a text description. Preserve relational constraints rather than
    # deriving a category from the held-out annotation's target box.
    concept = re.split(
        r"\s+(?:on|of|beside|next|behind|in|at|held|worn|carried|being|standing|sitting|seated|walking|held|to|directly|immediately|that|which|who)\b",
        target,
        maxsplit=1,
        flags=re.I,
    )[0]
    concept = re.sub(r"^(?:the|a|an)\s+", "", concept, flags=re.I)
    if concept.lower().startswith("what"):
        # For relational action questions, SAM proposes people and objects; the
        # complete question remains the verifier's source of target semantics.
        concept = (
            "person"
            if re.search(r"\b(man|woman|person|musician)\b", question)
            else "object"
        )
    query = (
        target
        if target != question.rstrip("?")
        else f"The target referred to by this question: {question}"
    )
    return query, concept
