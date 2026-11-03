"""Generation accounting and callback-compatible model calls."""
from collections.abc import Callable
import inspect

from .errors import InvalidScoreResponse, LocalBudgetExceeded


def charge_generation(usage: dict, stage: str, limit: int = 0) -> None:
    """Charge before execution; resolver/answer have separate, unbudgeted stages."""
    search = stage not in {'resolver', 'answer'}
    count = usage.get('search_generations', usage.get('local_score_calls', 0))
    if search and limit and count >= limit:
        raise LocalBudgetExceeded('Physical search-generation budget exhausted')
    if search:
        usage['search_generations'] = count + 1
    if stage != 'resolver':
        usage['local_score_calls'] = usage.get('local_score_calls', 0) + 1
    usage['physical_generations'] = usage.get('physical_generations', 0) + 1
    counts = usage.setdefault('generation_counts', {})
    counts[stage] = counts.get(stage, 0) + 1


def invoke_counted(method: Callable, *args, on_request: Callable, **kwargs):
    """Native scorers charge every generation; legacy injected scorers charge once."""
    parameters = inspect.signature(method).parameters
    accepts = 'on_request' in parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values())
    if accepts:
        return method(*args, on_request=on_request, **kwargs)
    on_request()
    return method(*args, **kwargs)


def generate_rating(generate: Callable, parse: Callable, prompt: str,
                    on_request: Callable, repair_hint: str):
    """Parse at most two generations, charging before each physical attempt."""
    replies = []
    for attempt in range(2):
        on_request()
        text = generate(prompt)
        replies.append(text)
        try:
            return parse(text), text, prompt
        except ValueError:
            if attempt:
                raise InvalidScoreResponse(replies) from None
            prompt += '\n' + repair_hint
    raise AssertionError('Unreachable generation retry state')
