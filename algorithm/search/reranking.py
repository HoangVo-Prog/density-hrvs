"""Conservative duplicate grouping and bounded comparative candidate verification.

Geometry proposes duplicate hypotheses; it cannot establish semantic identity.
All original boxes remain in the pool for audit and coverage evaluation.
"""
from __future__ import annotations

import json
import time

from ..core.calls import invoke_counted
from ..core.errors import InvalidScoreResponse, LocalBudgetExceeded
from ..models.comparison import build_comparison_sheet


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    union = area_a+area_b-intersection
    return intersection/union if union else 0.


def same_duplicate_hypothesis(a, b):
    # Keep part/owner roles and distinct concepts separate, including nested boxes.
    if (a.get('role', 'entity'), a.get('label', '')) != (b.get('role', 'entity'), b.get('label', '')):
        return False
    aa, bb = a['bbox'], b['bbox']
    if overlap(aa, bb) < .9:
        return False
    return all(min(aa[i+2]-aa[i], bb[i+2]-bb[i]) / max(aa[i+2]-aa[i], bb[i+2]-bb[i], 1) >= .9
               for i in (0, 1))


def group_candidates(proposals):
    groups = []
    for index, proposal in enumerate(proposals):
        group = next((g for g in groups if same_duplicate_hypothesis(proposal, proposals[g['representative']])), None)
        if group is None:
            groups.append({'representative': index, 'members': [index]})
        else:
            group['members'].append(index)
    return groups


def shortlist_candidates(proposals, groups, limit=5):
    # Preserve the inference top-1; trade off original rank and spatial novelty.
    remaining = list(range(len(groups)))
    chosen = []
    while remaining and len(chosen) < limit:
        def priority(index):
            rank = groups[index]['representative']
            novelty = 1-max((overlap(proposals[rank]['bbox'], proposals[groups[j]['representative']]['bbox'])
                             for j in chosen), default=0.)
            return .5/(rank+1) + .5*novelty
        selected = max(remaining, key=priority)
        chosen.append(selected)
        remaining.remove(selected)
    return [groups[index]['representative'] for index in chosen]


def rerank_regions(state, result):
    if not state.settings.get('comparative_reranking', False):
        return
    proposals = result['ranked_regions']
    groups = group_candidates(proposals)
    indices = shortlist_candidates(proposals, groups, state.settings.get('rerank_candidate_limit', 5))
    audit = {'status': 'pending', 'groups': groups, 'shortlist_pool_indices': indices,
             'original_pool_size': len(proposals), 'selected_pool_index': None}
    result['reranking'] = audit
    method = getattr(state.local_scorer, 'compare_candidates', None)
    if not indices or not callable(method) or state.remaining_final_generations() == 0:
        audit['status'] = 'no_candidates' if not indices else 'unavailable_or_budget'
        return
    selected = [proposals[i] for i in indices]
    sheet, mappings = build_comparison_sheet(state.source_image, selected,
        max_side=state.settings.get('verification_max_side', 1536))
    audit['evidence_mapping'] = mappings
    folder = state.artifacts/'reranking' if state.artifacts is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        sheet.save(folder/'input.png')
    tick = time.perf_counter()
    try:
        assessment = invoke_counted(method, sheet, state.query, len(indices),
            on_request=lambda: state.charge_final('comparative_verification'))
    except (LocalBudgetExceeded, InvalidScoreResponse) as exc:
        audit['status'] = type(exc).__name__
        audit['raw_responses'] = getattr(exc, 'responses', [])
        if folder is not None:
            (folder/'results.json').write_text(json.dumps(audit, indent=2)+'\n')
        return
    finally:
        state.record_cost('score', time.perf_counter()-tick)
    audit['assessment'] = assessment
    winner = assessment['best_index']
    rating = assessment['ratings'][winner]
    accepted = (not assessment['ambiguous'] and rating['answer'] == 'Yes'
                and rating['score'] >= state.settings.get('candidate_accept_score', .5)
                and rating.get('model_id') not in {None, 'not-run'})
    # An explicit comparative rejection/ambiguity supersedes earlier isolated scores.
    for proposal in proposals:
        proposal['comparative_eligible'] = False
    if accepted:
        pool_index = indices[winner-1]
        best = proposals[pool_index]
        best['previous_verifier'] = best.get('verifier')
        best['verifier'] = rating
        best['candidate_verified'] = True
        best['comparative_eligible'] = True
        audit['selected_pool_index'] = pool_index
        audit['status'] = 'selected'
        result['ranked_regions'] = [best]+[p for i,p in enumerate(proposals) if i != pool_index]
    else:
        audit['status'] = 'no_unambiguous_match'
        result['abstention'] = {'answer': 'Uncertain', 'reason': 'comparative_verification_failed'}
    if folder is not None:
        (folder/'results.json').write_text(json.dumps(audit, indent=2)+'\n')
