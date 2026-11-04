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


OBJECTS_PROMPT = """Extract SAM3 object prompts from the original question.
Include BOTH the requested physical object/part (role target) AND every reference
object used to identify it (role reference). Each object is a SEPARATE SAM3 call.
Keep known identifying colors and attributes, never invent an unknown answer.
Use short noun phrases without spatial clauses. Keep locations/relations in
context. Include an owning person/object as a reference when it identifies a
requested part. Do not return color/shape as an object. Never omit the target.
Return only JSON with target (description) and objects (list of role, prompt,
context). Role is target or reference. Always include at least one target.
Example question: What color is the cargo bed of the pickup diagonally above and
to the left of the turquoise car on the bottom approach?
Output: {"target":"cargo bed of pickup","objects":[{"role":"target","prompt":"pickup cargo bed","context":"diagonally above-left of turquoise car"},{"role":"reference","prompt":"pickup","context":"owns requested cargo bed"},{"role":"reference","prompt":"turquoise car","context":"bottom approach"}]}
Example question: What shape is the white structure in the green area at the bottom-left of the intersection?
Output: {"target":"white structure","objects":[{"role":"target","prompt":"white structure","context":"inside bottom-left green area"},{"role":"reference","prompt":"green area","context":"bottom-left of intersection"}]}
Question: """


def parse_object_plan(text):
    payload = _json_payload(text)
    if not isinstance(payload, dict) or not isinstance(payload.get('objects'), list):
        raise ValueError('Object plan must include an objects list')
    target = payload.get('target')
    if not isinstance(target, str) or not target.strip() or len(target)>240:
        raise ValueError('Missing physical target description')
    if not 1 <= len(payload['objects']) <= 16:
        raise ValueError('Expected 1 through 16 object prompts')
    objects=[];seen=set();counts={'target':0,'reference':0}
    for obj in payload['objects']:
        if not isinstance(obj, dict) or obj.get('role') not in counts:
            raise ValueError('Every object needs target or reference role')
        role=obj['role'];prompt=obj.get('prompt');context=obj.get('context','')
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>160 or len(prompt.split())>16:
            raise ValueError('Invalid object detector prompt')
        if not isinstance(context,str) or len(context)>500:
            raise ValueError('Invalid object context')
        prompt=' '.join(prompt.split()).strip(' .');key=(role,prompt.casefold(),context.casefold())
        if key in seen:continue
        counts[role]+=1;objects.append({'id':f'{role}_{counts[role]}','role':role,'prompt':prompt,'context':context.strip()});seen.add(key)
    if not counts['target']:
        raise ValueError('SAM3 prompt plan omitted the requested object')
    return {'target':target.strip(),'objects':objects,'references':[o for o in objects if o['role']=='reference']}
