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
| 7 | Optional same-instance consensus experiment | Screened on both 134-case paired view experiments; fewer exact correct answers and 2x answer requests; left off |

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
legacy 49/134 versus marked detail 43/134; among the 45 originally localized questions,
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

Qwen3-VL 8B and memory staging findings:
Coexistence test: running Qwen3-VL-8B-Instruct (4-bit NF4) alongside DINOv3 and SAM3
exhausts the 12 GiB GPU memory when all reside simultaneously in VRAM. To resolve this,
`model_residency="staged"` was introduced in `algorithm/models/residency.py`, which moves
each model to CUDA when actively called and returns it to CPU afterwards.

Evaluation results:
1. Full 134-question evaluation on frozen baseline candidate crops:
   - Comparing 2B vs 8B: exact answer matches jumped from 33/134 to 65/134, and abstentions
     plummeted from 38 down to 15 when using legacy crop view.
   - Marked answer views continued to regress relative to legacy crop views even with 8B,
     confirming visual synthetic boxes distract the VLM; legacy crop view remains primary.
2. Official dataset support (`data_official_official`):
   - Created dataset manifest adapter supporting the 501-image / 1,060-question format
     with direct ZIP streaming from `images.zip`.
   - Verified load and execution on image `000.jpg` with staged Qwen3-VL 8B; completed
     all iterations and generated predictions within GPU memory limits.
   - All 31 regression and invariant unit tests pass.


## Larger verifier and requested official export (2026-11-04)

The official 8B checkpoint is available in the secondary-drive cache. Explicit model ID,
revision, cache path and NF4 quantization are now configurable and saved with each answer;
no output from an 8B scorer is labelled 2B. The comparison tools use the same frozen
inference boxes and both answer views for either model. The 134-case paired experiment
`qwen8_answer_full` is complete: exact normalized complete answers are legacy 65/134,
marked-detail 57/134, versus 2B legacy 33/134 and marked 29/134. Legacy abstentions decrease
38 -> 15; marked 51 -> 22. This is an answer-stage deployment comparison (8B NF4 versus
2B bfloat16), not a clean parameter-count ablation or new end-to-end search result. The
literal normalized phrase diagnostic is 49 -> 65 for legacy and 43 -> 57 for marked.
The earlier 48 legacy diagnostic used a slightly different normalization; use the
consistent normalized-word-boundary comparison for both models. Exact scores are not
interchangeable with the inherited semantic rubric (baseline 53/134).

Standalone 8B NF4 peak allocation is about 6.15 GiB. Keeping 8B plus both FP32 vision
models resident fails with GPU OOM. `model_residency=staged` instead moves each model to
the GPU for its call and offloads it afterward, preserving weights, input resolution,
prompts and physical generation accounting. A real SAM3+DINO+8B smoke completes with
legacy answering (0411_q02 -> green). This establishes feasible execution; its owner box
still misses the tiny badge annotation. Tests now cover staging cleanup, model-independent
answer evidence, both dataset formats and exact three-digit ZIP filenames (31 tests).

The requested `data_official_official` contains 501 images and 1,060 questions in an
image-record annotation export. The reader projects exactly image ID, question ID and
question into inference; answers, box attributes, points and targets are exposed only to
post-prediction evaluation. Its three-digit filenames are matched exactly before the
historic four-digit fallback. Inspection labels use the actual scorer/backend identity.

Content fingerprint and exact question-text checks find all 67 audited development images
and 134 questions in this export. The additional group has 434 images and 926 questions.
Membership is frozen in `algo1_improve/official_dataset_membership.json`; report the groups
separately. A first-two-image native pilot was started with 8B NF4, staged residency, legacy
answer view, 32 physical search generations, full default DINO/SLIC resolution and all
unproven new query/part/reranking/fallback features off. One case completed successfully; the pilot was stopped at its saved checkpoint to
finish the 8B comparative-reranking development experiment before the long official run.
Full evaluation and the native controller matrix remain pending. The selected model configuration is based on the
completed development comparison, not on the new dataset's answers.


Same-instance two-view consensus screening (`model_comparison.json`) includes 127 original
inference-accepted owners. With 8B, legacy answering produces 65 exact correct answers
among 118 non-abstentions; agreement-only consensus produces 52 among 79 non-abstentions,
at twice the answer-generation cost. With 2B, the corresponding exact counts are 33 -> 19.
The two views share pixels and are not independent votes. Consensus remains off.

Long-run execution support: `algorithm.tools.run_experiment` hashes dataset contents and
configuration, saves executable source plus an archive, and always executes/resumes the
verified frozen copy. Working-branch edits cannot silently change a running experiment.
One-worker GPU batches now run serially so interruption exits cleanly rather than waiting
for thread-pool work to drain. Per-question completion records include elapsed time,
physical/search generations, primary detector and feature calls, and peak GPU allocation.
32 regression tests pass, including source/config resume safety and dataset-input isolation.
