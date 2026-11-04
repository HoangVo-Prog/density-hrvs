"""Question-only reference extraction, preserving known attributes and roles."""
import json
from ..models.scoring import _json_payload

REFERENCE_PROMPT = """Extract the physical REFERENCE objects that identify the requested target.
Return ONLY JSON: {"target":"requested physical object or part", "references":[{"prompt":"short noun phrase", "context":"location or relationship stated in the question"}]}.
Each prompt will be sent to SAM3 SEPARATELY. Keep known identifying colors and
attributes in the prompt, e.g. turquoise car, yellow bus, person in orange shirt.
Do not put spatial clauses or Locate instructions in the SAM3 prompt. Put these
in context instead. Do not invent an unknown target color or answer the question.
The requested target and its direct owning object are NOT external references.
If there are separate external references, include each, including intermediate
objects in chained relations. Do not combine several objects into one prompt.
If there is no external reference but the question asks about a part or an object
worn/held by an identifiable owner, that owner can be the reference.
Named physical landmarks such as a crosswalk are references. Pure directions or
abstract areas such as upper-right, center, traffic or neighborhood are not objects.
Return an empty references list if the question contains no reference object.
Examples:
What color is the cargo bed of the pickup diagonally above and to the left of the turquoise car on the bottom approach?
{"target":"cargo bed of pickup","references":[{"prompt":"turquoise car","context":"on the bottom approach; target diagonally above and left"}]}
What color is the car above the dark gray car directly left of the pale yellow car?
{"target":"car","references":[{"prompt":"dark gray car","context":"target is above it; directly left of pale yellow car"},{"prompt":"pale yellow car","context":"dark gray car is directly left of it"}]}
What color are the trousers of the person in white walking beside the person in a light jacket?
{"target":"trousers of person in white","references":[{"prompt":"person in a light jacket","context":"beside target's owner"}]}
What color is the long bus on the lower-left side?
{"target":"long bus","references":[]}
Question: """


def parse_reference_plan(text):
    plan = _json_payload(text)
    if not isinstance(plan, dict):
        raise ValueError('Reference plan must be an object')
    target = plan.get('target')
    refs = plan.get('references')
    if not isinstance(target, str) or not target.strip() or len(target) > 240:
        raise ValueError('Missing physical target description')
    if not isinstance(refs, list) or len(refs) > 12:
        raise ValueError('references must be a list with at most 12 objects')
    result = []; seen = set()
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError('Each reference must be an object')
        prompt, context = ref.get('prompt'), ref.get('context', '')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 160 or len(prompt.split()) > 16:
            raise ValueError('Invalid reference detector prompt')
        if not isinstance(context, str) or len(context) > 500:
            raise ValueError('Invalid reference context')
        prompt = ' '.join(prompt.split()).strip(' .')
        if prompt.casefold() not in seen:
            result.append({'id': f'reference_{len(result)+1}', 'prompt': prompt, 'context': context.strip()})
            seen.add(prompt.casefold())
    return {'target': target.strip(), 'references': result}
