# algo1/improve implementation status

The authoritative baseline is commit `1ce161b`. Existing benchmark artifacts are preserved in
`/media/tinhanhnguyen/sub/outputs/official_benchmark_full`.

## Scope and evidence

| Phase | Required behavior | Status |
|---|---|---|
| 0 | Frozen source/JSON evidence, independently checked metrics, diagnostic selection | Verified on all 134 questions; 408 evidence/source files frozen and hashed |
| 1 | Fallback cap, physical generation accounting, acceptance eligibility, partial evidence | Implemented; 10 targeted tests pass |
| 2 | Marked context and clean enlarged answer detail; frozen-box GPU comparison | Implemented; 12 tests pass; full 134-case paired GPU experiment complete; opt-in |
| 3 | Structured entity/part/anchor resolver and linked boxes | Implemented opt-in; 19 tests pass; native GPU probes complete; no quality gain established |
| 4 | Instance-aware grouping and comparative reranking | Implemented opt-in; 23 tests pass; 134-case GPU evaluation complete; 41 localized versus baseline 45; keep off |
| 5 | Bounded crop recovery and Grounding DINO fallback | Bounded fallback implemented opt-in; 26 tests pass; native detector and integration probes complete; expanded-crop retry pending |
| 6 | Matched-budget controller and partition ablations | BFS/spatial controls implemented; 28 tests pass; native matched-budget matrix pending |
| 7 | Optional same-instance consensus experiment | Pending |

Each stage will be committed separately after relevant checks. Implementation and measured model
quality are tracked separately; no phase is validated by tests of another phase. Keep all 134
audited questions as development/regression data and reserve unseen images for final claims.

Baseline metrics: top-1 IoU passes 45/134; retained-pool passes 80/134; strict answer passes
50/134; semantic passes 53/134; joint semantic+IoU passes 26/134. The 54 pool-coverage failures
are not all part-box mismatches. The 48 search requests plus one answer are not an off-by-one bug.

## Reproduce baseline freeze

```bash
python -m algorithm.tools.experiment_baseline \
  /media/tinhanhnguyen/sub/outputs/official_benchmark_full \
  /media/tinhanhnguyen/sub/outputs/algo1_improve/baseline
```

The command validates every top/best IoU from exported boxes, freezes JSON evidence and source,
records SHA-256 hashes, and writes a label-free diagnostic selection including all 26 joint passes.
Calling it again verifies the snapshot rather than overwriting it.

Verified snapshot: `/media/tinhanhnguyen/sub/outputs/algo1_improve/baseline`; diagnostic selection: 37 questions, including all 26 joint passes. GPU: RTX 3080 Ti, 12 GB, accessible outside the sandbox.

Phase 1: fallback 0/1/3 and remaining-budget checks pass; rejected/unchecked candidates cannot answer; repairs charge before each generation; partial completed checks survive interruption without node reward backup. Behavioral comparison switches: `require_verified_answer` and `strict_fallback_limit`. Physical accounting remains accurate for every configuration. GPU model imports verified (CUDA available).

Phase 2 GPU pilot: same frozen boxes and same model/runtime for both views, on 0288_q01,
0411_q01, 0719_q01 and 2783_q01. Marked detail changes the backpack answer from Uncertain
to green (label requires Neon green), trousers from Uncertain to beige (label Khaki),
and white trousers from blue to white. The existing red-shirt case remains correct.
These observations are not a full accuracy estimate. The tool's exact-complete-answer
metric penalizes sentences and is deliberately conservative; do not equate it with the
saved semantic rubric. Peak allocated GPU memory was approximately 4.3 GiB.
Pilot artifacts: `/media/tinhanhnguyen/sub/outputs/algo1_improve/phase2_pilot`.
The new answer view is switchable with `answer_view=legacy_crop|marked_detail`.

Full Phase 2 experiment: `/media/tinhanhnguyen/sub/outputs/algo1_improve/phase2_full`.
All 134 questions were run with frozen existing top boxes, both views, the same cached Qwen
revision, and the same runtime. Exact-complete-answer diagnostic: legacy 33/134 versus
marked detail 29/134. An additional literal normalized GT-phrase containment diagnostic:
legacy 48/134 versus marked detail 43/134; among the 45 originally localized questions,
25/45 versus 24/45. These diagnostics are not the inherited semantic rubric and require
adjudication for synonyms and sentence wording. The new view increased abstentions (38 to 51).
It improves particular part-reading cases but regresses others; `answer_view=legacy_crop`
therefore remains the default. No overall accuracy improvement is claimed. Structured part
binding and further controlled evidence experiments remain necessary.


Phase 3: `structured_query=true` extracts entity, requested part and reference descriptions
from the question; the original full relation description is retained. `part_refinement=true`
runs one bounded SAM3 pass around a verified owner and verifies a cyan part against its
magenta owner. Entity and attribute boxes remain separate; the primary localization box
is never swapped using evaluation labels. Finalization requests share the search budget,
with a configurable reserve. Standalone SAM3 loading now uses the pinned local checkpoint
instead of an unavailable sibling-project import.

GPU evidence: the simpler flat resolver parsed 8/8 questions in one request each, but
unknown-object/support role assignments are still imperfect. On four frozen inference
owners, SAM3 produced Qwen-accepted backpack, badge and phone parts; the camera lower dome
had no part proposal. Acceptance is not ground truth: the backpack verifier hallucinated
vegetables. Full relational answering still abstained on the three accepted parts. A
focused-question probe also failed to establish an improvement (badge blue, backpack and
phone Uncertain, camera dome black). Focused rewriting is therefore not enabled.
One full SAM3+DINO+Qwen pipeline smoke on 0411_q02 completed within its 12-generation
search cap, but found no eligible owner. These are implementation/plumbing checks, not
accuracy gains. Artifacts are under `algo1_improve/phase3_query_probe_v2`,
`phase3_part_probe` and `phase3_smoke` on the secondary output drive. Both features remain
off by default until comparative candidate verification and broader evaluation support them.


Phase 4: groups only near-identical geometric hypotheses (IoU >= .9, similar dimensions,
same role/concept), preserving all members and nested part/owner boxes. This is conservative
duplicate grouping, not proof of semantic identity. A rank/diversity shortlist of at most
five representatives is compared using numbered global context and clean enlarged detail
crops without confidence/label hints. All-No, Uncertain, or ambiguous comparisons produce
no answer-eligible winner. Earlier isolated Yes scores cannot override a comparative
rejection. Comparative repairs consume the shared finalization reserve. Original proposal
pools are preserved for coverage evaluation. The experimental basis is visual marking
(Set-of-Mark, https://arxiv.org/abs/2310.11441), not evidence that this 2B model must improve.

The first native schema failed three initial cases; those artifacts are retained in
`phase4_frozen_pool`. Explicit JSON syntax fixed the four-case pilot (`phase4_frozen_pool_v2`):
three selections and one no-unambiguous-match. No accuracy conclusion follows from that.
Full paired frozen-pool inference is running in `phase4_full`. The feature remains off by
default pending localization and regression analysis on all 134 cases.


Completed Phase 4 result (`phase4_full/evaluation.json`): 134 frozen-pool questions;
113 selections, 18 ambiguous/rejected comparisons, one empty pool, two malformed responses.
Selected localization IoU >= .5: 41/134 versus original 45/134, zero gains and four regressions.
The five representative shortlist retains an IoU-qualified box in 66/134 versus 80/134 in
full pools and 67/134 in original top-five pools. Display index 1 dominates predictions;
counterbalanced ordering and larger-model testing are needed before any benefit claim.
The feature remains disabled by default.

Phase 5: pinned Grounding DINO tiny revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`,
loaded lazily from an explicit cache. Trigger only on an empty root, a child of a visually
positive parent, or a missing requested part. Default cap is one fallback detector call
per question, shared across these triggers; no remaining verifier budget means no new
fallback work. Raw pre-threshold scores, cap decisions and source crop coordinates are
saved. Box-only rectangular masks are explicitly labelled; Grounding DINO confidence is
never represented as SAM3 confidence. Qwen remains responsible for accepting proposals.
A global preflight makes root misses reachable because the original tree partitions root
before evaluation. Existing nonempty SAM3 root proposals are currently cached rather than
introduced as extra verified proposals.

Native GPU detector probes on frozen inference inputs: 0467_q02 scooter 2 boxes; 0411_q02
badge 2; 0649_q02 phone 2; 1497_q02 lower dome 4. Counts establish execution, not accuracy.
The first badge/phone proposals visibly cover their small objects; false proposals also
exist. Full unchanged-model-stack smoke on 0467_q02 completes (no fallback triggered because
current root SAM3 is nonempty), but does not localize the evaluation target. A controlled
empty-SAM integration uses native Grounding DINO and Qwen, with RGB features solely for
plumbing; it is explicitly not a detector-quality benchmark. Expanded overlapping crop
retry and broad accuracy evaluation remain pending. All fallback features stay opt-in.


Phase 6 support: `strategy=bfs|greedy|uct|agent-uct` and
`partition_mode=spatial|semantic`. BFS uses the globally shallowest active frontier and
then node ID; spatial mode uses configurable overlapping quadrants and does not load or
invoke DINO. Default behavior stays semantic Agent-UCT. Feature encoder and primary/fallback
detector calls are recorded separately from physical Qwen generations. Tests establish
selection order and absence of DINO calls in spatial mode, not algorithm superiority.
Native comparisons must hold the model version, generation cap, proposal/verification
settings, partition depth and answer/reranker configuration fixed and report wall time
alongside all model-call counts. No UCT usefulness claim is supported yet.

User-authorized larger-model experiment: official Qwen3-VL-8B-Instruct (the family has 8B,
not 7B), pinned revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, downloaded to
`/media/tinhanhnguyen/sub/model_cache`. An isolated NF4 double-quantized bfloat16 GPU smoke
answers a synthetic red image correctly, with peak allocation about 6.15 GiB. This confirms
standalone feasibility, not task quality or coexistence with every other loaded model.
