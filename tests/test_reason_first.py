import pytest
from algorithm.models.reason_first import parse_reason_first
from algorithm.models.spatial_context import spatial_answer_prompt, spatial_verification_reason_prompt

def test_response_order_and_decision():
    result=parse_reason_first('{"answer":"Yes","reason":"Pickup and bus are visible."}', 'Yes')
    assert list(result)==['reason','answer']
    with pytest.raises(ValueError,match='changed'):
        parse_reason_first('{"reason":"Unclear vehicle","answer":"No"}', 'Yes')

@pytest.mark.parametrize('raw',['{}','{"reason":"","answer":"Yes"}','{"reason":"Visible","answer":3}'])
def test_invalid_response(raw):
    with pytest.raises(ValueError):parse_reason_first(raw)

def test_prompts_keep_reason_before_answer():
    meta={'original_image_size':[100,100],'region_bbox_xyxy':[0,0,100,100],'reasoned_json':True}
    for p in [spatial_answer_prompt('Color?',meta), spatial_verification_reason_prompt('Color?',meta,'Yes')]:
        assert p.index('"reason"')<p.index('"answer"')
        assert '12 words' in p
