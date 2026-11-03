"""Validated question-only entity, attribute-region and reference descriptions."""
from __future__ import annotations

import json


def _phrase(value, field: str, maximum: int = 160) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f'{field} must be a nonempty phrase of at most {maximum} characters')
    return ' '.join(value.split())


def parse_query_plan(text: str, question: str | None = None) -> dict:
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```':
            raise ValueError('Incomplete query-plan JSON fence')
        text = '\n'.join(lines[1:-1])
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError('Query plan must be an object')
    if 'entity_concept' in payload:
        # A flat generation schema is less fragile for the local 2B model.
        # Build internal identifiers ourselves instead of asking it to cross-reference them.
        references = payload.get('reference_concepts', [])
        if not isinstance(references, list):
            raise ValueError('reference_concepts must be a list')
        lower = (question or '').lower()
        attribute = next((name for name in ('color', 'pattern', 'shape', 'material')
                          if lower.startswith('what '+name)), 'identity')
        part = payload.get('part_concept')
        if part is not None:
            part = _phrase(part, 'part_concept')
        constraints = payload.get('scene_constraints', [])
        if not isinstance(constraints, list):
            raise ValueError('scene_constraints must be a list')
        payload = {'entity': {'concept': payload['entity_concept']},
            'attribute_region': {'concept': part, 'relation': 'associated with the selected entity'} if part else None,
            'requested_attribute': attribute,
            'anchors': [{'id': f'anchor_{i}', 'concept': value} for i,value in enumerate(references,1)],
            'relations': [],
            'scene_constraints': [value for value in constraints
                                  if isinstance(value,str) and (not question or value.lower() in lower)]}
    entity = payload.get('entity')
    if not isinstance(entity, dict):
        raise ValueError('Query plan needs an entity')
    result = {'entity': {'id': 'entity', 'concept': _phrase(entity.get('concept'), 'entity.concept')},
              'attribute_region': None, 'requested_attribute': _phrase(payload.get('requested_attribute'), 'requested_attribute', 40),
              'anchors': [], 'relations': [], 'scene_constraints': []}
    part = payload.get('attribute_region')
    if part is not None:
        if not isinstance(part, dict):
            raise ValueError('attribute_region must be an object or null')
        result['attribute_region'] = {'id': 'attribute', 'concept': _phrase(part.get('concept'), 'attribute_region.concept'),
                                      'relation': _phrase(part.get('relation'), 'attribute_region.relation')}
    anchors = payload.get('anchors', [])
    if not isinstance(anchors, list) or len(anchors) > 8:
        raise ValueError('anchors must contain at most eight objects')
    ids = {'entity'} | ({'attribute'} if part is not None else set())
    for index, anchor in enumerate(anchors, 1):
        if not isinstance(anchor, dict):
            raise ValueError('anchor must be an object')
        identifier = _phrase(anchor.get('id', f'anchor_{index}'), 'anchor.id', 40)
        if identifier in ids:
            raise ValueError('Duplicate entity identifier')
        ids.add(identifier)
        result['anchors'].append({'id': identifier, 'concept': _phrase(anchor.get('concept'), 'anchor.concept')})
    relations = payload.get('relations', [])
    if not isinstance(relations, list) or len(relations) > 12:
        raise ValueError('relations must contain at most twelve objects')
    for relation in relations:
        if not isinstance(relation, dict) or relation.get('subject') not in ids or relation.get('object') not in ids:
            raise ValueError('Relation references an unknown entity')
        result['relations'].append({'subject': relation['subject'], 'object': relation['object'],
                                   'predicate': _phrase(relation.get('predicate'), 'relation.predicate', 80)})
    constraints = payload.get('scene_constraints', [])
    if not isinstance(constraints, list) or len(constraints) > 8:
        raise ValueError('scene_constraints must contain at most eight phrases')
    result['scene_constraints'] = [_phrase(value, 'scene_constraint') for value in constraints]
    if question is not None:
        # Keep the complete wording for ownership/spatial checks, including multi-hop relations.
        result['relation_description'] = question
    return result


STRUCTURED_QUERY_PROMPT = """Extract search roles from the question. Do not answer it.
Return ONE JSON object with exactly these four fields:
entity_concept: short detector phrase for the entity to identify.
part_concept: the requested clothing/belonging/subpart to inspect, or null.
reference_concepts: list of short phrases for other objects used as clues.
scene_constraints: list of location phrases copied exactly from the question, or [].

For clothing or belongings, entity_concept is the owning PERSON with identifying
clothing and part_concept is the requested item. For a standalone object,
entity_concept is that object and part_concept is null unless a subpart is asked.
For a question asking WHAT something is, preserve the unknown object descriptively
without guessing its identity. If asked what supports another object, the entity
is its unknown support, while the supported object is a reference. Do not replace
an unknown hanging object with the railing from which it hangs.
Use only information present in the question. Keep entity phrases under 16 words.
The full question will remain available for all spatial and ownership checks.
Question: {question}
JSON:"""
