"""Test real logits routing and continuous probability backup without loading weights."""
import math
from types import SimpleNamespace
import pytest
import torch
from PIL import Image
from algorithm.models.yes_no_probability import yes_no_token_ids, score_yes_no_logits
from algorithm.models.spatial_context import build_region_context
from algorithm.models import qwen_scoring
from algorithm.search.pipeline import evaluate_node
from algorithm.search.eligibility import answer_eligible
from test_question_region import state_with, node


class Tokenizer:
    def encode(self, label, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [{"Yes": 1, "No": 2}[label]]


def test_binary_probability_is_not_full_vocabulary_or_a_generated_score():
    r = score_yes_no_logits(torch.tensor([12., 2., 1.]), {"Yes": 1, "No": 2})
    assert r["score"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert r["yes_probability_full_vocabulary"] < .001
    assert r["score_source"] == "yes_no_token_probability"
    assert r["answer"] == "Yes" and r["token_logits"] == {"Yes": 2., "No": 1.}


def test_no_and_tie_keep_continuous_probability():
    r = score_yes_no_logits(torch.tensor([0., 1., 2.]), {"Yes": 1, "No": 2})
    assert r["answer"] == "No" and r["score"] == pytest.approx(.26894142137)
    tie = score_yes_no_logits(torch.zeros(3), {"Yes": 1, "No": 2})
    assert tie["score"] == .5 and tie["answer"] == "Uncertain"
    with pytest.raises(ValueError, match="finite"):
        score_yes_no_logits(torch.tensor([0., float("nan"), 1.]), {"Yes": 1, "No": 2})


def test_token_labels_must_be_single_and_distinct():
    assert yes_no_token_ids(Tokenizer()) == {"Yes": 1, "No": 2}
    with pytest.raises(ValueError, match="one next token"):
        yes_no_token_ids(SimpleNamespace(encode=lambda *a, **kw: [1, 2]))
    with pytest.raises(ValueError, match="distinct"):
        yes_no_token_ids(SimpleNamespace(encode=lambda *a, **kw: [1]))


@pytest.mark.parametrize("spatial", [True, False])
def test_public_verifier_reads_one_forward_pass_and_never_generates_json(monkeypatch, spatial):
    seen = []; forwards = []; charges = []
    class Batch(dict):
        def to(self, device):
            return self
    class Processor:
        tokenizer = Tokenizer()
        image_processor = SimpleNamespace(size={})
        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages)
            return Batch(input_ids=torch.tensor([[4, 5]]))
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, **kwargs):
            forwards.append(kwargs)
            return SimpleNamespace(logits=torch.tensor([[[7., 2., 1.]]]))
        def generate(self, **kwargs):
            raise AssertionError("Verifier must not call generate()")
    monkeypatch.setattr(qwen_scoring.AutoProcessor, "from_pretrained", lambda *a, **kw: Processor())
    monkeypatch.setattr(qwen_scoring.Qwen3VLForConditionalGeneration, "from_pretrained", lambda *a, **kw: Model())
    scorer = qwen_scoring.load_qwen_scorer("cpu")
    detail = Image.new("RGB", (20, 20), "red")
    context = build_region_context(Image.new("RGB", (100, 100)), (10, 10, 30, 30), node_id=3)
    kwargs = {"spatial_context": context} if spatial else {}
    r = scorer.score_question_region(detail, "What color is the car?", on_request=lambda: charges.append(1), **kwargs)
    assert len(charges) == len(forwards) == len(seen) == 1
    assert forwards[0]["logits_to_keep"] == 1 and forwards[0]["use_cache"] is False
    assert r["score"] == pytest.approx(.73105857863)
    assert r["score_source"] == "yes_no_token_probability"
    content = seen[0][0]["content"]
    images = [x["image"] for x in content if x["type"] == "image"]
    assert len(images) == (2 if spatial else 1)
    assert images[-1].tobytes() == detail.tobytes()
    assert "Answer Yes or No only" in r["prompt"]
    assert '"score":0-100' not in r["prompt"]
    if spatial:
        assert r["spatial_metadata"]["region_bbox_xyxy"] == [10, 10, 30, 30]


def test_no_probability_is_not_zeroed_before_search_backup():
    def verifier(im, q, *, on_request):
        on_request()
        return {"answer": "No", "score": .4, "score_source": "yes_no_token_probability", "model_id": "mock"}
    s = state_with(lambda *a: [], verifier)
    evidence = evaluate_node(s, node())
    assert evidence["raw_reward"] == .4
    assert evidence["proposals"][0]["score"] == .4
    assert not answer_eligible(evidence["proposals"][0], s.settings)
