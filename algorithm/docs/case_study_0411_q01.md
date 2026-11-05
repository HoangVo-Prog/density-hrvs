# Success case: 0411_q01

This case is from `outputs/official_part1_resolver/0411_q01/`. It is a clean
localization success: the top-ranked box is also the best box and reaches IoU
0.957 against the annotated target. It also shows the target resolver, the
spatial fallback, recursive zoom, and a local-budget stop in one run.

## Why this case

- The question asks for a **shirt color**, but the resolver changes the detector
  target to the **person operating the TV camera**. This is exactly the intended
  resolver behavior: localize the person, then let the verifier inspect the
  shirt in context.
- The first pass finds a plausible box, but the recursive child re-detects the
  same object at a tighter scale and improves both detector confidence and IoU.
- The search finds the correct top-1 box before the local generation budget runs
  out, and the interrupted final evaluation does not pollute the ranking.
- The artifact trail is complete: root split, both accepted parent evaluations,
  both winning evidence sheets, the tree trace, and post-run ground-truth
  inspection.

## Inputs and audit-only ground truth

| Item | Value |
| --- | --- |
| Image | `0411`, 4269 x 2846 px |
| Question | What color is the shirt worn by the person operating the TV camera on the upper balcony? |
| Search query | `the shirt worn by the person operating the TV camera on the upper balcony` |
| Heuristic detector concept | `shirt` (would be produced by `question_target`) |
| Resolved detector prompt | `the person operating the TV camera` |
| Annotated answer | `Red` |
| Annotated target box | `[2342, 1810, 2418, 1923]` |

The answer and target box are loaded only after inference by
`reporting/inspection.py`. They are audit data, not model inputs.

## Outcome

| Metric | Value |
| --- | --- |
| Status | complete |
| Stop reason | `local_budget` |
| Elapsed | 104.56 s |
| API calls / reported cost | 0 / $0.00 |
| Local Qwen image generations | 8 |
| Local Qwen text generations (resolver) | 1 |
| Cache hits | 0 |
| Candidate checks | 8 |
| Verification path | per-candidate context/detail sheets, stop at first clear `Yes` |
| Gated regions (no SAM3 candidates) | 8 |
| Iterations in trace | 14 (13 complete, 1 interrupted) |
| Tree nodes | 63 |
| Ranked proposals | 27 |
| Verified proposals | 2 (N4 candidate 1, N19 candidate 1) |
| Top-1 IoU | **0.9574136008918618** |
| Best IoU | **0.9574136008918618** |
| Top-1 = best | yes |

Relevant controls: `strategy=agent-uct`, `max_depth=3`, `branching_factor=5`,
`candidate_check_limit=2`, `candidate_accept_score=0.5`, `sam3_threshold=0.1`,
`max_local_scores=8`, `stop_on_match=False`.

The parent `summary.json` reports 27 of 78 attempted questions with top-1 IoU at
least 0.5 for this run. This case is one successful localization, not a held-out
accuracy claim.

## Step-by-step

### 0. Resolve the detector target

`tools/run.py` derives the query from the question. The regex heuristic in
`question_target()` would extract `shirt` from "the shirt worn by...". The local
Qwen resolver, called once by `prepare_run()`, instead returns:

```text
the person operating the TV camera
```

That phrase is stored in `config.sam_prompt` and used for SAM3. The complete
query remains the verifier description, so Qwen still judges the shirt and the
upper-balcony relationship. The text-only resolver call is recorded separately
from the image-generation budget.

### 1. Root split and spatial fallback

The root is the full 4269 x 2846 image. DINOv3 -> SLIC -> K-means produced five
semantic groups, but one group had occupancy 0.247, below the default limit of
0.25. `_split_quality()` therefore chose the spatial fallback, producing four
quadrants:

| Node | Bbox | Role |
| --- | --- | --- |
| N1 | `[0, 0, 2134, 1423]` | top-left |
| N2 | `[2134, 0, 4269, 1423]` | top-right |
| N3 | `[0, 1423, 2134, 2846]` | bottom-left |
| N4 | `[2134, 1423, 4269, 2846]` | bottom-right, contains the target |

This is a useful reminder that "semantic split" is conditional. The root's
first level is spatial in this run because the semantic candidate geometry was
poor, not because semantic splitting is disabled.

### 2. The first pass finds a plausible box

Iterations 1-4 visit all four unvisited root children, in order of their
estimated remaining cost. N1, N2, and N3 return reward 0. N4 returns 0.8017.

N4's evaluation:

- 4 SAM3 boxes after NMS;
- one Qwen call on candidate 1's two-panel context/detail sheet;
- candidate 1 answered `Yes` at 0.95, so the check loop stopped before
  candidate 2;
- one accepted candidate;
- box `[2340, 1811, 2419, 1906]`;
- SAM3 confidence 0.5709;
- fused candidate score 0.8363;
- IoU against the annotated box **0.8137**.

The saved run uses the per-candidate verification path: each checked box gets a
CONTEXT/TARGET-DETAIL sheet (`candidate_001_evidence.png`), not a whole-region
numbered-box sheet. The artifacts contain no `region_candidates.png` or
`region_candidate_scores.json`. The saved config omits `verification_mode`; the
artifact set and candidate-check counts identify the effective path. The early
`Yes` is why N4 cost one generation even though it had four distinct boxes.

This is already a plausible hit. The box is 95 px tall, while the annotated box
is 113 px tall: it covers the upper part of the target but undershoots the lower
part. The search does not stop, because `stop_on_match` defaults to `False`, and
the branch has a high reward relative to the zero-reward siblings.

### 3. Zoom into the promising quadrant

N4 is split using the same semantic pipeline. Its five semantic groups again
fail the geometry check (one aspect ratio is 9.24), so the children are four
spatial quadrants. The relevant child is:

```text
N19 = [2134, 1423, 3201, 2134]
```

At iteration 5, UCT selects the N4 branch again. The trace terms are computed at
the root for N4, while the evaluated leaf is N19:

```text
quality             = 0.8017   (N4 average reward)
exploration bonus   = 1.7761
estimated cost      = 8.9278 s
cost penalty        = 0.1786
selection index     = 2.3992
```

N4 has only one visit and the root has four, so the exploration bonus is large
enough to outweigh the small cost penalty. N19 is unvisited, so it is chosen
over the other frontier leaves.

### 4. The recursive child improves the box

N19's evaluation:

- 6 SAM3 boxes after NMS;
- one Qwen call on candidate 1's context/detail sheet;
- candidate 1 answered `Yes` at 1.00, so the loop stopped before candidate 2;
- one accepted candidate;
- box `[2341, 1809, 2419, 1924]`;
- SAM3 confidence 0.8936;
- fused candidate score 0.9681;
- IoU against the annotated box **0.9574**.

The measured improvement from the first pass to the recursive child is:

| Quantity | N4 first pass | N19 child | Change |
| --- | --- | --- | --- |
| SAM3 confidence | 0.5709 | 0.8936 | +0.3227 |
| Qwen score | 0.95 | 1.00 | +0.05 |
| Fused candidate score | 0.8363 | 0.9681 | +0.1318 |
| Box height | 95 px | 115 px | +20 px |
| IoU | 0.8137 | 0.9574 | +0.1437 |

Both boxes are around the same upper-body/shirt region. The child crop gives
SAM3 a tighter, more confident detection and the fused candidate ranks first.
N19 becomes the top-ranked proposal.

### 5. Continue, then stop on the local budget

After N19, the search still visits other frontiers:

- N20, N5, N10, N15, N21, N22, N16, and N6 all return 0;
- the 8 gated regions are N1, N2, N20, N10, N15, N22, N16, and N6; N3, N5,
  and N21 had SAM3 candidates but no accepted candidate;
- 8 image generations have been consumed by candidate verification.

Iteration 14 selects N11. Before the next Qwen generation,
`LocalBudgetExceeded` is raised. The step is recorded with `completed = False`
and `raw_reward = None`, and no reward is backed up. The search stops with
`local_budget`. This is why the last trace row is incomplete but the ranking is
still valid.

N19 was expanded after its completed evaluation and produced five semantic
children (N23-N27). None of them were evaluated before the budget ended, so the
run is a budget-limited best effort, not an exhaustive search.

### 6. Final ranking

The final report contains 13 evaluated clusters and 27 proposals. Sorting by
candidate score puts N19 candidate 1 first:

```text
rank 1: N19  score 0.9681  sam3 0.8936  bbox [2341, 1809, 2419, 1924]
          Qwen Yes, 1.00, IoU 0.9574

rank 2: N4   score 0.8363  sam3 0.5709  bbox [2340, 1811, 2419, 1906]
          Qwen Yes, 0.95, IoU 0.8137
```

The remaining proposals are rejected, unchecked, or zero-score boxes from other
regions. Top-1 IoU equals best IoU, so the ranking did not hide a better box in
this run.

## Trace table

The `quality`, `exploration_bonus`, and `selection_index` columns are the
**top-level branch** terms recorded by `_select_path()`. The `leaf` and
`reward` columns describe the frontier that was actually evaluated. This is why
iteration 5 shows a quality of 0.8017 while evaluating N19.

| Iter | Leaf | Depth | Branch Q | Bonus | Est. cost (s) | Index | Leaf reward | Complete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N1 | 1 | 0.0000 | 0.0000 | 14.6801 | -0.2936 | 0.0000 | yes |
| 2 | N2 | 1 | 0.0000 | 0.0000 | 13.2383 | -0.2648 | 0.0000 | yes |
| 3 | N3 | 1 | 0.0000 | 0.0000 | 12.9347 | -0.2587 | 0.0000 | yes |
| 4 | N4 | 1 | 0.0000 | 0.0000 | 8.9460 | -0.1789 | 0.8017 | yes |
| 5 | N19 | 2 | 0.8017 | 1.7761 | 8.9278 | 2.3992 | 0.8987 | yes |
| 6 | N20 | 2 | 0.8502 | 1.3251 | 4.9276 | 2.0768 | 0.0000 | yes |
| 7 | N5 | 2 | 0.0000 | 1.9529 | 8.6423 | 1.7801 | 0.0000 | yes |
| 8 | N10 | 2 | 0.0000 | 2.0188 | 8.7312 | 1.8442 | 0.0000 | yes |
| 9 | N15 | 2 | 0.0000 | 2.0752 | 8.7742 | 1.8997 | 0.0000 | yes |
| 10 | N21 | 2 | 0.5668 | 1.2265 | 3.6765 | 1.7198 | 0.0000 | yes |
| 11 | N22 | 2 | 0.4251 | 1.0840 | 3.0574 | 1.4479 | 0.0000 | yes |
| 12 | N16 | 2 | 0.0000 | 1.5605 | 4.9147 | 1.4622 | 0.0000 | yes |
| 13 | N6 | 2 | 0.0000 | 1.5854 | 5.0501 | 1.4844 | 0.0000 | yes |
| 14 | N11 | 2 | 0.0000 | 1.6082 | 5.0277 | 1.5076 | none | no |

## Reward arithmetic

For both accepted regions, the region reward is:

```text
combined = 0.65 * best_qwen
         + 0.20 * mean_sam3_confidence
         + 0.10 * count_score
         + 0.05 * diversity_score
```

N4:

```text
0.65 * 0.95
+ 0.20 * 0.5709
+ 0.10 * 0.20        # 1 verified box / cap 5
+ 0.05 * 1.00        # diversity = 1 - mean pairwise IoU; one box => 0 IoU
= 0.8017
```

N19:

```text
0.65 * 1.00
+ 0.20 * 0.8936
+ 0.10 * 0.20
+ 0.05 * 1.00
= 0.8987
```

The score is therefore consistent with the measured improvement in detector
confidence and verifier rating. It is still a heuristic: the single-verified-box
case gives `mean_pairwise_iou = 0` and `diversity = 1`, which is maximally
favorable, and the weights are hand-set.

Candidate fusion is separate:

```text
N4:  (0.7 * 0.95 + 0.3 * 0.5709) = 0.8363
N19: (0.7 * 1.00 + 0.3 * 0.8936) = 0.9681
```

## Why it worked

1. **The resolver changed the target to something the detector can find.**
   Detecting `shirt` directly is ambiguous in a crowded 4K scene. Resolving to
   "the person operating the TV camera" gives SAM3 a coherent object proposal.
   The full original query is still used by Qwen, so the shirt and relationship
   are not lost.
2. **The first level was coarse but sufficient to expose the right quadrant.**
   Even though N4's box undershot, its reward made it the only positive branch.
   UCT therefore revisited it rather than spending the next iteration on an
   unvisited zero-reward quadrant.
3. **Recursive re-encoding improved both detection and verification.**
   N19's SAM3 confidence rose from 0.5709 to 0.8936, and its box became a better
   fit for the annotated 113 px target. This is the clearest evidence for the
   value of the recursive split in this case.
4. **Early-stop verification kept the budget small.**
   The per-candidate loop checks in confidence order and stops at the first clear
   `Yes`. N4 and N19 each spent one Qwen generation on their top candidate, even
   though they had four and six distinct boxes respectively.
5. **Empty regions were gated.**
   Eight regions with no SAM3 candidates never reached Qwen. The run spent its
   local budget on candidate verification instead of empty regions.
6. **The budget stop did not corrupt the ranking.**
   The interrupted N11 evaluation was not backed up, and all previously
   completed evidence remained rankable.

## What this case does not show

- It does **not** show answer accuracy. The annotated answer is `Red`, and Qwen's
  reason mentions a red shirt, but the search output is a ranked box. The system
  does not emit or verify the final color answer.
- It is **not** a held-out benchmark result. It is one case from a development
  run, selected after the run completed.
- There is no resolver-off control for this query. The resolver's contribution is
  inferred from the saved prompt and the pipeline design, not measured by an
  ablation.
- The ground-truth box was not available to the model. The successful IoU is
  measured only after the fact by the inspection step.
- `local_budget` stopped the run while unvisited descendants remained. A larger
  budget might find a better box, or a worse one; this run does not prove that
  N19 is globally optimal.
- The Qwen scores 0.95 and 1.00 are ordinal judgments, not probabilities. The
  region reward is a hand-weighted heuristic.
- IoU 0.957 is a strong localization signal, but it does not prove that the
  verifier reasoned about the correct person for the right reason. The saved
  evidence sheet and reason text are the audit trail for that judgment.

## Artifacts to inspect

All links are relative to this file and go through the
`outputs/official_part1_resolver` symlink.

| Artifact | What to look at |
| --- | --- |
| [Tree and trace](../../outputs/official_part1_resolver/0411_q01/stages/tree/index.html) | All 14 iterations, branch selection, node expansion |
| [Root stages](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0000/index.html) | DINO PCA, SLIC history, K-means, spatial fallback |
| [N4 stages](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0004/index.html) | First successful pass; 4 boxes, 1 accepted |
| [N4 candidate evidence](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0004/candidate_001_evidence.png) | Context/detail sheet for the 0.8137 IoU box |
| [N19 stages](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0019/index.html) | Recursive region; 6 boxes, 1 accepted, semantic children |
| [N19 candidate evidence](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0019/candidate_001_evidence.png) | Context/detail sheet for the top-1 box |
| [N19 candidate input](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0019/candidate_001_input.json) | Exact bbox mapping and evidence-window coordinates |
| [N19 region score](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0019/region_score.json) | Qwen's full reason and raw response |
| [N19 reward components](../../outputs/official_part1_resolver/0411_q01/stages/nodes/node_0019/region_score_components.json) | Best Qwen, confidence, count, diversity, combined |
| [results.json](../../outputs/official_part1_resolver/0411_q01/run/results.json) | Tree, trace, ranked clusters, 27 proposals |
| [Proposal preview](../../outputs/official_part1_resolver/0411_q01/run/preview.jpg) | Ranked boxes drawn on the original image |
| [Inspection report](../../outputs/official_part1_resolver/0411_q01/inspection/inspection.json) | Annotated answer, target box, per-candidate IoU |
| [Predictions vs ground truth](../../outputs/official_part1_resolver/0411_q01/inspection/0411_predictions_vs_ground_truth.jpg) | Post-run localization audit |
| [Top-1 candidate view](../../outputs/official_part1_resolver/0411_q01/inspection/candidate_views/0411_candidate_001.jpg) | Inspection view of the winning box |
