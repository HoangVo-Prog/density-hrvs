import base64
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import threading
import urllib.request

from PIL import Image
from transformers import AutoTokenizer

from ..core.contracts import CandidatePrompt, RegionAssessment, VerifierScore
from .qwen_scoring import (
    ANSWER_PROMPT,
    REGION_CANDIDATES_PROMPT,
    SAM_PROMPT_REWRITE_PROMPT,
    TARGET_RESOLVER_PROMPT,
    clean_sam_prompt,
)
from .scoring import CONTEXT_INSTRUCTIONS, SCORING_PROMPT, ScoringView, _ignore_request

DEFAULT_ENDPOINT = "http://127.0.0.1:18380"
VLLM_MODELS = {
    "Qwen/Qwen3-VL-2B-Instruct": {
        "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "processor": {"shortest_edge": 65536, "longest_edge": 1048576},
    },
    "OpenGVLab/InternVL3-8B-hf": {
        "revision": "259a3b64a14623c0ec91a045cb43f7c5af5fa6af",
        "processor": "native checkpoint settings",
    },
}


def png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def yes_no_probability(logprob_yes: float, logprob_no: float) -> tuple[float, float]:
    if not (math.isfinite(logprob_yes) and math.isfinite(logprob_no)):
        raise ValueError(f"non-finite Yes/No logprobs: {logprob_yes}, {logprob_no}")
    top = max(logprob_yes, logprob_no)
    total = top + math.log(math.exp(logprob_yes - top) + math.exp(logprob_no - top))
    return math.exp(logprob_yes - total), math.exp(logprob_yes) + math.exp(logprob_no)


class VllmScorer:
    def __init__(self, model: str, endpoint: str = DEFAULT_ENDPOINT):
        self.endpoint = endpoint.rstrip("/")
        self.model_revision = VLLM_MODELS[model]["revision"]
        self.backend = "vllm"
        served = [m["id"] for m in self._get("/v1/models")["data"]]
        if model not in served:
            raise ValueError(f"server at {endpoint} serves {served}, not {model}")
        self.model_id = model
        tokenizer = AutoTokenizer.from_pretrained(model, revision=self.model_revision, local_files_only=True)
        ids = [tokenizer.encode(text, add_special_tokens=False) for text in ("Yes", "No")]
        assert all(len(i) == 1 for i in ids), ids
        self.yes_id, self.no_id = ids[0][0], ids[1][0]
        self.probability_forwards = 0
        self.http_requests = 0
        self._lock = threading.Lock()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.endpoint + path, timeout=30) as reply:
            return json.load(reply)

    def _chat(self, content: list[dict], max_tokens: int, **extra) -> dict:
        body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "top_p": 1,
            "repetition_penalty": 1,
            "max_tokens": max_tokens,
            **extra,
        }
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self._lock:
            self.http_requests += 1
        with urllib.request.urlopen(request, timeout=600) as reply:
            return json.load(reply)["choices"][0]

    def _text(self, prompt: str, max_tokens: int, image: Image.Image | None = None) -> str:
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image_url", "image_url": {"url": png_data_uri(image)}})
        return self._chat(content, max_tokens)["message"]["content"] or ""

    def rewrite(self, query: str, concept: str) -> str:
        target = concept.strip()
        if not target:
            return ""
        prompt = SAM_PROMPT_REWRITE_PROMPT.format(concept=target, query=json.dumps(query))
        return clean_sam_prompt(self._text(prompt, 64))

    def resolve(self, query: str) -> str:
        prompt = TARGET_RESOLVER_PROMPT.format(query=json.dumps(query))
        return clean_sam_prompt(self._text(prompt, 32))

    def answer(self, image: Image.Image, question: str, *, on_request: Callable[[], None] = _ignore_request) -> str:
        on_request()
        return self._text(ANSWER_PROMPT.format(question=json.dumps(question)), 96, image)

    def _probability(self, image, instruction, query, target) -> VerifierScore:
        prompt = instruction.replace(
            "First identify the boxed object and name it explicitly in reason.",
            "Assess only the cyan-boxed object.",
        ) + (
            f"\nSearch description: {json.dumps(query)}\n"
            f"Does {target} match the complete search description? "
            "Answer Yes only when all required attributes and relations are visibly supported. "
            "Otherwise answer No, including when evidence is insufficient. "
            "Return only Yes or No, without JSON, a numeric rating, or an explanation.\nAnswer:"
        )
        content = [
            {"type": "image_url", "image_url": {"url": png_data_uri(image)}},
            {"type": "text", "text": prompt},
        ]
        choice = self._chat(content, 1, logprobs=True, logprob_token_ids=[self.yes_id, self.no_id],
                            return_tokens_as_token_ids=True)
        first = choice["logprobs"]["content"][0]
        seen = {e["token"]: e["logprob"] for e in [first, *first["top_logprobs"]]}
        p_yes, mass = yes_no_probability(seen[f"token_id:{self.yes_id}"], seen[f"token_id:{self.no_id}"])
        with self._lock:
            self.probability_forwards += 1
        return {
            "score": p_yes, "p_yes": p_yes, "yes_no_mass": mass,
            "answer": "Yes" if p_yes >= 0.5 else "No",
            "reason": "Yes/No-normalized next-token probability; no self-rating generated.",
            "model_id": self.model_id, "model_revision": self.model_revision, "backend": self.backend,
            "prompt": prompt, "input_size": image.size,
        }

    def __call__(self, image: Image.Image, query: str, *, view_type: ScoringView = "region",
                 on_request: Callable[[], None] = _ignore_request) -> VerifierScore:
        instruction = CONTEXT_INSTRUCTIONS if view_type == "context_detail" else ""
        instruction += SCORING_PROMPT.split("Return ONLY one JSON object", 1)[0]
        target = "the cyan-boxed target" if view_type == "context_detail" else "this image"
        on_request()
        return self._probability(image, instruction, query, target)

    def score_candidates(self, image: Image.Image, query: str, candidates: list[CandidatePrompt], *,
                         on_request: Callable[[], None] = _ignore_request) -> RegionAssessment:
        instruction = REGION_CANDIDATES_PROMPT.split("Rank the candidates explicitly.", 1)[0]
        instruction += "\nCandidates:\n" + json.dumps(candidates, indent=2)
        on_request()
        with ThreadPoolExecutor(max_workers=5) as pool:
            scores = pool.map(lambda c: self._probability(image, instruction, query, f"numbered candidate {c['index']}"), candidates)
            ratings = {c["index"]: s for c, s in zip(candidates, scores)}
        ranking = sorted(ratings, key=lambda index: ratings[index]["score"], reverse=True)
        best = ranking[0]
        ambiguous = ratings[best]["score"] < 0.5 or (
            len(ranking) > 1 and ratings[best]["score"] - ratings[ranking[1]]["score"] < 0.05
        )
        return {"ratings": ratings, "ranking": ranking, "best_index": best, "ambiguous": ambiguous}
