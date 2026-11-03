# algo1/improve implementation status

The authoritative baseline is commit `1ce161b`. Existing benchmark artifacts are preserved in
`/media/tinhanhnguyen/sub/outputs/official_benchmark_full`.

## Scope and evidence

| Phase | Required behavior | Status |
|---|---|---|
| 0 | Frozen source/JSON evidence, independently checked metrics, diagnostic selection | Verified on all 134 questions; 408 evidence/source files frozen and hashed |
| 1 | Fallback cap, physical generation accounting, acceptance eligibility, partial evidence | Implemented; 10 targeted tests pass |
| 2 | Marked context and clean enlarged answer detail; frozen-box GPU comparison | Implemented; 12 tests pass; four-case paired GPU pilot complete |
| 3 | Structured entity/part/anchor resolver and linked boxes | Pending |
| 4 | Instance-aware grouping and comparative reranking | Pending |
| 5 | Bounded crop recovery and Grounding DINO fallback | Pending |
| 6 | Matched-budget controller and partition ablations | Pending |
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
