"""Separate answer eligibility from proposal availability and detector objectness."""


def answer_eligible(proposal: dict, config: dict) -> bool:
    if not config.get('require_verified_answer', True):
        return True
    rating = proposal.get('verifier', {})
    return bool(
        rating.get('answer') == 'Yes'
        and rating.get('model_id') not in {None, 'not-run'}
        and rating.get('score', 0) >= config.get('candidate_accept_score', .5)
        and proposal.get('candidate_verified', True)
    )


def eligible_regions(proposals: list[dict], config: dict) -> list[dict]:
    return [p for p in proposals if answer_eligible(p, config)]
