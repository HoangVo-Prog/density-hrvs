"""Recoverable local-model and budget failures, separate from execution errors."""


class InvalidScoreResponse(RuntimeError):
    """A local verifier failed schema validation; retain its raw replies for audit."""

    def __init__(self, responses: list[str]):
        super().__init__(
            "Local Qwen produced invalid JSON ratings after repair; region skipped"
        )
        self.responses = responses


class APIBudgetExceeded(RuntimeError):
    """The next physical request would exceed the configured search budget."""


class LocalBudgetExceeded(APIBudgetExceeded):
    """The next local generation would exceed its separate compute budget."""
