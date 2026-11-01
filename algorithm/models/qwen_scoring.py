"""Local, pinned Qwen3-VL-2B-Instruct visual query-match scoring."""

from collections.abc import Callable
import json
from typing import Any, Protocol, cast

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..core.contracts import CandidatePrompt, RegionAssessment, VerifierScore
from ..core.errors import InvalidScoreResponse
from .scoring import (
    CONTEXT_INSTRUCTIONS,
    SCORING_PROMPT,
    ScoringView,
    _ignore_request,
    parse_region_assessment,
    parse_verifier_score,
)

QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"

SAM_PROMPT_REWRITE_PROMPT = """You convert a visual search description into one short phrase for an open-vocabulary object detector.
Use this pattern exactly: "the {concept} of the <person or object with identifying clothing>".
Include the articles. Keep the identifying clothing color. Drop location and action details.
Use at most 10 words. Do not answer the question. Return only the phrase, with no quotes.

Search description: {query}
Phrase:"""


TARGET_RESOLVER_PROMPT = """You are the localization-target resolver for a visual search system.
Given a search description, choose the ONE object or region the detector should localize.
Rules:
- If the description asks about an attribute (color, pattern, material) of a person's clothing or body part, return the person or upper body, not the attribute noun.
- If the description asks about a standalone object (umbrella, cart, clock, bag, camera), return that object.
- Keep identifying clothing only when it helps disambiguate the person.
- Return one short noun phrase, at most 8 words, with no quotes and no explanation.
Examples:
- "the shoes of the man in the black polo shirt" -> "the man in the black polo shirt"
- "the shirt worn under the vest of the boy" -> "the boy"
- "the face of the clock on the wall" -> "the clock"
- "the umbrella held over the boy" -> "the umbrella"
Search description: {query}
Localization target:"""


REGION_CANDIDATES_PROMPT = """You are verifying several candidate object boxes inside one clustered image region.
The image shows the whole region. Each SAM3 candidate is drawn as a numbered box and labelled with its concept and confidence.
Evaluate every numbered candidate against the complete search description.
Judge visible evidence only. All required attributes and spatial relations must belong to the same object.
Do not choose a candidate only because its SAM3 confidence is high.
Use the surrounding region to check relations and nearby reference objects.
If required evidence is too small or obscured, mark that candidate Uncertain.
Rank the candidates explicitly. Do not give identical scores unless the boxes are truly indistinguishable.

Candidates:
{candidates}

Return ONLY one JSON object:
{{"ranking":[3,1,5,2,4],"best_index":3,"candidates":[{{"index":1,"answer":"Yes","score":92,"reason":"..."}}]}}
ranking must list every candidate index exactly once, best first.
best_index must equal ranking[0].
answer is exactly "Yes", "No", or "Uncertain". score is a match rating from 0 to 100.
reason is a short visual explanation as a JSON string.

Search description: {query}
"""


class LocalScorer(Protocol):
    """Scores ordinary crops or marked context/detail sheets using local weights."""

    def __call__(
        self,
        image: Image.Image,
        query: str,
        *,
        view_type: ScoringView = "region",
        on_request: Callable[[], None] = _ignore_request,
    ) -> VerifierScore: ...


def clean_sam_prompt(text: str) -> str:
    """Return one short, unquoted detector phrase, or '' when malformed."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    text = text.splitlines()[0].strip() if text else ""
    text = text.strip(" \t\r\n\"'`")
    if text.endswith("."):
        text = text[:-1].rstrip()
    text = " ".join(text.split())
    if not text or len(text) > 120 or len(text.split()) > 16:
        return ""
    if text.lower().startswith(("the ", "a ", "an ")):
        return text
    return f"the {text}"


def load_qwen_scorer(device: str = "auto") -> LocalScorer:
    """Load cached weights once; returned callable performs no network requests."""
    selected = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL, revision=QWEN_REVISION, local_files_only=True
    )
    # Bound visual tokens to keep DINOv3 and the scorer resident on a 12GB GPU.
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 1048576}
    # Transformers' generation mixin stubs omit attributes supplied at runtime.
    model = (
        cast(
            Any,
            Qwen3VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL,
                revision=QWEN_REVISION,
                local_files_only=True,
                dtype=torch.bfloat16 if selected.startswith("cuda") else torch.float32,
                attn_implementation="sdpa",
            ),
        )
        .to(selected)
        .eval()
    )

    def _generate_text(prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(selected)
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        return processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

    def _generate_image_text(
        image: Image.Image, prompt: str, max_new_tokens: int
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(selected)
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        return processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

    @torch.inference_mode()
    def rewrite(query: str, concept: str) -> str:
        target = concept.strip()
        if not target:
            return ""
        prompt = SAM_PROMPT_REWRITE_PROMPT.format(
            concept=target, query=json.dumps(query)
        )
        return clean_sam_prompt(_generate_text(prompt, 64))

    @torch.inference_mode()
    def resolve(query: str) -> str:
        prompt = TARGET_RESOLVER_PROMPT.format(query=json.dumps(query))
        return clean_sam_prompt(_generate_text(prompt, 32))

    @torch.inference_mode()
    def score(
        image: Image.Image,
        query: str,
        *,
        view_type: ScoringView = "region",
        on_request: Callable[[], None] = _ignore_request,
    ) -> VerifierScore:
        prompt = (
            (CONTEXT_INSTRUCTIONS if view_type == "context_detail" else "")
            + SCORING_PROMPT
            + json.dumps(query)
        )
        replies: list[str] = []
        on_request()
        for attempt in range(2):
            text = _generate_image_text(image, prompt, 256)
            replies.append(text)
            try:
                rating = parse_verifier_score(text)
                return {
                    **rating,
                    "model_id": QWEN_MODEL,
                    "raw_response": text,
                    "prompt": prompt,
                    "input_size": image.size,
                    "model_revision": QWEN_REVISION,
                }
            except ValueError:
                if attempt:
                    raise InvalidScoreResponse(replies) from None
                prompt += '\nReturn only valid JSON, for example {"answer":"Uncertain","score":30,"reason":"Required detail is unclear."}.'
        raise RuntimeError("Local Qwen scoring failed")

    @torch.inference_mode()
    def score_candidates(
        image: Image.Image,
        query: str,
        candidates: list[CandidatePrompt],
        *,
        on_request: Callable[[], None] = _ignore_request,
    ) -> RegionAssessment:
        """Assess and rank the numbered SAM3 boxes on the whole cluster image."""
        prompt = REGION_CANDIDATES_PROMPT.format(
            candidates=json.dumps(candidates, indent=2),
            query=json.dumps(query),
        )
        expected = [candidate["index"] for candidate in candidates]
        replies: list[str] = []
        on_request()
        for attempt in range(2):
            text = _generate_image_text(image, prompt, 512)
            replies.append(text)
            try:
                assessment = parse_region_assessment(text, expected)
                return {
                    "ranking": assessment["ranking"],
                    "best_index": assessment["best_index"],
                    "ambiguous": assessment["ambiguous"],
                    "ratings": {
                        index: {
                            **rating,
                            "model_id": QWEN_MODEL,
                            "raw_response": text,
                            "prompt": prompt,
                            "input_size": image.size,
                            "model_revision": QWEN_REVISION,
                        }
                        for index, rating in assessment["ratings"].items()
                    },
                }
            except ValueError:
                if attempt:
                    raise InvalidScoreResponse(replies) from None
                prompt += "\nReturn only valid JSON with a full ranking and every candidate index exactly once."
        raise RuntimeError("Local Qwen candidate scoring failed")

    setattr(score, "rewrite", rewrite)
    setattr(score, "resolve", resolve)
    setattr(score, "score_candidates", score_candidates)
    return cast(LocalScorer, score)
