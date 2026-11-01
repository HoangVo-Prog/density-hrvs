# Algorithm walkthrough

This document follows one search from request preparation to the final ranked
regions. The implementation is deliberately split into pure stages so each
decision can be inspected in isolation.

## 0. Problem statement

**Input**

- an RGB image (`uint8`, or floating point in `[0, 1]`);
- a search description, such as `the shirt worn by the person operating the TV camera`;
- a `SearchConfig` (validated defaults are in `search/config.py:8`);
- optionally injected `scorer`, `detector`, `feature_extractor`, or
  `local_scorer` callables.

**Output**

`run_visual_search()` (`search/runner.py:41`) returns a `SearchResult`
(`core/contracts.py:357`) containing:

- the image-region tree and allocation trace;
- ranked clusters and ranked SAM3 proposal boxes;
- actual usage counters (HTTP calls, local generations, cache hits);
- a stop reason and any recoverable warnings.

**Non-goals**

- The search does not generate the final textual answer. It localizes the
  requested object so a downstream stage can crop it and answer.
- The verifier score is an ordinal visual judgment, not a calibrated probability.
- The tree is a bounded heuristic search, not an exhaustive proof that the
  globally best box was found.

## 1. Request preparation and target resolution

`tools/run.py` first calls `question_target()` (`core/query.py:4`) to strip a
leading "What color/pattern/shape is..." wrapper. The remainder becomes the
search description. For example:

```text
question: What color is the shirt worn by the person operating the TV camera on the upper balcony?
query:    the shirt worn by the person operating the TV camera on the upper balcony
heuristic detector concept: shirt
```

The heuristic concept is only a fallback. `prepare_run()`
(`search/setup.py:191`) then:

1. validates the config and rejects an empty query;
2. converts the image to RGB;
3. loads the local Qwen verifier unless a custom scorer was injected;
4. asks Qwen to resolve the **localization target**;
5. loads DINOv3 and SAM3 unless they were injected;
6. creates a mutable `RunState` with caches, budgets, and artifact paths.

### Target resolver

`_resolve_sam_prompt()` (`search/setup.py:28`) sends the full description to
the local Qwen scorer. The prompt in `TARGET_RESOLVER_PROMPT`
(`models/qwen_scoring.py:34`) tells the model to choose the object the detector
should localize:

- for an attribute of a person's clothing or body part, return the person or
  upper body, not the attribute noun;
- for a standalone object, return that object;
- return one short noun phrase.

The resolver is independent of the initial noun parsed by `question_target`.
For the example above, it changes the detector concept from `shirt` to:

```text
the person operating the TV camera
```

This is important because SAM3 is more reliable at proposing a person/upper body
than at proposing an abstract attribute region such as "shirt". The complete
query is still used later for verification.

If the resolver fails or returns an empty phrase, the original detector prompt is
kept and `sam_prompt_rewrite_failed` is emitted. The resolver is called once per
run. Its text-only generation is counted in `usage["local_prompt_calls"]`; in the
current implementation it does not consume `max_local_scores`.

`SearchRequest` and `CandidateEvidence` are frozen dataclasses. `RunState`
(`search/runtime.py:65`) owns all mutable per-run data, including:

- `score_cache`, `detection_cache`, `checked_cache`, `split_cache`;
- observed clusters/proposals and deduplication state;
- observed stage timings for cost estimation;
- the usage ledger and artifact destination.

## 2. Search tree

The root node is the whole image:

```text
bbox = (0, 0, width, height)
mask = all True
depth = 0
```

`explore_tree()` creates the root, expands it immediately, and starts iterating
(`search/tree.py:564`). Each `SearchNode` adds search statistics to a
`ChildRegion`:

| Field | Meaning |
| --- | --- |
| `bbox` | tight rectangle in original-image coordinates |
| `mask` | active pixels inside that rectangle; local to the bbox |
| `children` | node IDs created by one split |
| `visits` | completed evaluations backed up through this node |
| `reward_sum` | sum of raw rewards backed up through this node |
| `evaluated` | this node's own evaluation finished |
| `exhausted` | no unevaluated descendants remain within the depth limit |

Masks are local to the node bbox. A child mask always excludes pixels outside its
parent's active mask, so every split is a strict reduction of the active search
space. Child bboxes can still overlap because a semantic cluster can be spatially
fragmented; the masks are what partition pixels.

If the root cannot be split at all, `explore_tree()` clears the root's
`exhausted` flag so the root is evaluated once instead of returning an empty
result. Otherwise the tree is finite because of:

- `max_depth` (default 3);
- `min_region_pixels` (default 256 active pixels);
- the minimum crop side check (`< 32` pixels);
- `max_iterations` and the two budgets.

## 3. Branch selection: cost-aware UCT

The selector is a visual-search adaptation of Agent-UCT. A completed evaluation
returns the same raw reward regardless of how deep the leaf is; the cost term only
affects **selection** of the next branch.

For each active child `i` of a parent:

```text
Q_i = reward_sum_i / visits_i                     (0 when unvisited)
U_i = c * sqrt(log(N_parent + 1) / visits_i)      (0 when unvisited)
C_i = remaining_cost(i)                           (forecast of unpaid work)
P_i = lambda * C_i / T_ref                        (agent-uct only)
index_i = Q_i + U_i - P_i
```

Defaults:

```text
c = exploration_weight = 1.4
lambda = cost_weight = 0.2
T_ref = reference_cost_seconds = 10.0
```

Strategy behavior (`_branch_terms`, `search/tree.py:100`):

| Strategy | Selection index |
| --- | --- |
| `agent-uct` | `Q + U - P` |
| `uct` | `Q + U` |
| `greedy` | `Q` |

Selection (`select_child`, `search/tree.py:160`) works as follows:

1. Collect unfinished children (`not exhausted`).
2. If any child has never been visited, restrict the pool to unvisited children.
3. Score every child in the pool.
4. Choose the maximum index; ties go to the lowest node ID.

Unvisited quality is unknown. The cost forecast may therefore order initial
visits, but it cannot exclude an unvisited branch. In the UCT and greedy
strategies all unvisited children tie at index `0`, so the lowest ID is visited
first.

### Remaining-cost forecast

`remaining_cost()` (`search/tree.py:75`) recursively averages the estimated cost
of active descendants:

```text
if no active children:
    return max(0, estimate(node))

weights = (child.visits + 1/n_active) / (sum(child.visits) + 1)
return sum(weights * remaining_cost(child) for child in active)
```

The `1/n_active` pseudo-visit makes an unvisited set a simple average instead of
a division by zero. `estimate_node()` (`search/pipeline.py:323`) prices the work
that has not yet been cached:

- detection cost, unless the node is already in `detection_cache`;
- one whole-region Qwen call in `region` mode if candidates are pending and the
  node has not been checked;
- up to `candidate_check_limit` Qwen calls in `candidate` mode;
- split cost, if the node is below `max_depth` and not in `split_cache`.

Before the first sample of a stage, the estimator uses 1.0 s for detection,
3.0 s for scoring, and 1.0 s for splitting. After that it uses the observed mean.
The mean is floored at 1 ms.

### Path selection and trace semantics

`_select_path()` (`search/tree.py:246`) walks from the root to a frontier leaf,
calling `select_child()` at every expanded level. It records the selection terms
from the **first level only**. The trace row's `node_id` is the leaf, but its
`quality`, `exploration_bonus`, and `selection_index` belong to the top-level
branch that was selected after the root.

This distinction matters when reading a trace: on iteration 5 of the case study,
`quality = 0.802` is the average reward of root child N4, while the evaluated
leaf is the deeper node N19 with reward `0.899`.

## 4. Node evaluation

`evaluate_node()` (`search/pipeline.py:310`) does the following:

1. Sets the current node/folder in `RunState.context`.
2. Crops the **rectangular** region `rgb[y0:y1, x0:x1]`.
3. Saves the exact crop and a full-scene overlay.
4. Chooses the SAM3-first path (`sam3_first=True`, the default) or the
   legacy whole-region path.

The evaluation crop is not masked. The node mask is used to build children and
to render DINO/SLIC inputs, not to black out pixels before SAM3 or Qwen. The
candidate evidence sheets are cropped from the clean original image.

### 4.1 SAM3 proposals

`detect_region()` (`search/evaluation.py:132`):

- runs the detector on the crop with `settings["sam_prompt"] or query`;
- keeps detections with confidence strictly greater than `sam3_threshold`
  (default 0.1);
- keeps at most `max_proposals_per_node` (default 16) before mask resizing;
- sorts by confidence descending;
- caches the resulting list per node.

`distinct_detections()` (`search/region_scoring.py:24`) applies confidence-ordered
NMS. A lower-confidence box is suppressed when its IoU with any kept box is
greater than `region_nms_iou` (default 0.7). Every retained detection must have a
finite confidence in `[0, 1]`.

If the NMS list is empty, the region is **gated**
(`search/pipeline.py:247`):

- Qwen is not called;
- the raw reward is zero;
- `usage["gated_regions"]` is incremented;
- the event records `gate = "no_candidates"`.

This is a deliberate compute saving and is the reason empty regions can be
explored cheaply.

### 4.2 Qwen verification

Two verification modes exist.

**`region` mode (default).** `verify_region_candidates()`
(`search/candidates.py:229`) takes the top `region_candidate_limit` boxes
(default 5), draws each as a numbered box on the whole cluster crop, and sends
one image to Qwen. The `REGION_CANDIDATES_PROMPT`
(`models/qwen_scoring.py:50`) asks for one JSON rating per candidate:

```json
{"candidates":[{"index":1,"answer":"Yes","score":92,"reason":"..."}]}
```

The parser requires every expected candidate index exactly once. If the local
scorer does not implement `score_candidates`, the code falls back to per-candidate
context/detail sheets.

**`candidate` mode.** `verify_candidates()` (`search/candidates.py:205`) checks
candidates in confidence order, up to `candidate_check_limit` (default 2), and
stops at the first clear `Yes` at or above `candidate_accept_score` (default 0.5).
Each candidate gets a context/detail evidence sheet built by
`build_proposal_evidence()` (`models/evidence.py:92`):

- a CONTEXT panel with at least 384 px of padding and at least three target
  widths/heights of surrounding scene;
- a TARGET DETAIL panel with a small margin, enlarged to inspect the object;
- a cyan box for the same target in both panels;
- a 1536 x 1024 canvas by default.

This mode gives Qwen more local detail but costs one generation per checked
candidate. The default region mode is cheaper: one generation for several boxes.

### 4.3 Candidate and region reward

A candidate is **verified** when Qwen answers `"Yes"` and its normalized score is
at least `candidate_accept_score`.

Candidate fusion (`candidate_score`, `search/region_scoring.py:48`):

```text
fused = (0.7 * qwen_score + 0.3 * sam3_score) / (0.7 + 0.3)
```

A `"No"` candidate contributes `qwen_score = 0` to the fused score.

The region reward (`aggregate_verified_region`,
`search/region_scoring.py:69`) is:

```text
best_qwen    = max Qwen score among checked candidates whose answer != "No"
verified     = candidates with answer == "Yes" and score >= accept
confidence   = mean SAM3 confidence of verified boxes
count        = min(len(verified), region_box_count_cap) / region_box_count_cap
overlap      = mean pairwise IoU among verified boxes
diversity    = 1 - overlap, or 0 when there are no verified boxes
combined     = (0.65 * best_qwen
              + 0.20 * confidence
              + 0.10 * count
              + 0.05 * diversity) / 1.0
```

The defaults sum to 1, but the implementation normalizes by the actual weight
sum. If there are no verified boxes, `combined = 0`, regardless of the other
terms.

Gate values explain a zero or partial reward:

| Gate | Meaning |
| --- | --- |
| `no_candidates` | no distinct SAM3 boxes remain after thresholding and NMS |
| `budget` | candidates existed but none were checked |
| `unverified` | candidates were checked but none met the acceptance rule |
| `verified` | at least one checked candidate was accepted |

The reward is intentionally multi-term, but it is still a heuristic:

- `best_qwen` lets one strong candidate dominate;
- with a single verified box, `mean_pairwise_iou = 0` and `diversity = 1`, so the
  diversity term is maximally favorable;
- count saturates at `region_box_count_cap = 5`;
- the score is not calibrated and should not be read as a probability.

### 4.4 Global proposal registration

`register_proposals()` (`search/pipeline.py:60`) deduplicates by:

```text
(global bbox, hash of exact proposal crop and query)
```

If the same identity appears again, the higher-scoring copy replaces the stored
one. All proposals, including rejected and unchecked ones, remain in
`state.observed_regions` for final ranking.

## 5. Splitting a region

After a completed evaluation, the node is expanded unless a stop rule fired.
`split_node()` (`search/pipeline.py:35`) caches one split per node. The heavy
work is in `semantic_children()` (`search/partitioning.py:424`).

### 5.1 Stop-before-split rules

`_should_skip_split()` returns no children when:

- `depth >= max_depth`;
- active pixels `< min_region_pixels`;
- the smaller crop side is `< 32` pixels.

A skipped split marks the node exhausted and emits `split_skipped`.

### 5.2 DINOv3 features

The current crop is re-encoded from scratch with DINOv3:

- `region_pixels()` renders the crop with pixels outside the node mask set to
  black;
- DINOv3 runs at up to `feature_max_side` (default 1024), aligned to the patch
  size;
- the patch-token features are reduced to three PCA components;
- each component is clipped at the 1st/99th percentiles and mapped to `[0, 1]`;
- the three-channel map is resized back to the crop with nearest-neighbor
  interpolation.

This is a **visualization-scale PCA color map**, not a text-aligned embedding.
Parent encodings are not reused because the child crop is a new, higher-resolution
view. The map is saved as `dino_pca.png` and `dino_pca.npz`.

### 5.3 SLIC with entropy selection

`optimize_slic_n_segments()` (`models/regions.py:160`) tries the requested counts
(default `200, 400, 600, 800, 1000, 1500`). For each candidate it:

1. runs SLIC on the PCA color map;
2. quantizes each color channel into `entropy_bins` bins (default 8);
3. computes the per-superpixel color entropy;
4. records mean, p90, and max entropy.

The candidate with the lowest **mean entropy** wins. Ties favor fewer requested
segments because the candidates are processed in ascending order. The selected
labels are saved with the full SLIC history in `slic_metrics.json`.

### 5.4 K-means grouping

`cluster_slic_regions()` (`models/regions.py:262`) computes the mean PCA color of
each superpixel and, optionally, its normalized centroid:

```text
feature = [mean_PCA_r, mean_PCA_g, mean_PCA_b,
           spatial_weight * centroid_x,
           spatial_weight * centroid_y]
```

Defaults: `n_clusters = min(branching_factor, number_of_superpixels)` (at most
5), `spatial_weight = 0.75`, `random_state = 42`, `n_init = 10`.

K-means assigns each superpixel to a cluster. Every pixel inherits its
superpixel's cluster. The cluster labels are resized back to the original crop
with nearest-neighbor interpolation.

### 5.5 Child construction and quality gate

`_children_from_clusters()` intersects each cluster with the parent's active mask
and discards empty or full-parent clusters. Each child is tightened to its mask
bounds and records the original SLIC IDs it contains.

`_split_quality()` (`search/partitioning.py:331`) falls back to four spatial
quadrants when any of these is true:

| Reason | Test |
| --- | --- |
| `too_few_children` | fewer than two children |
| `no_zoom` | every child bbox is at least 90% of the parent area |
| `poor_geometry` | occupancy `< 0.25`, aspect `> 6`, or largest connected component `< 0.7` of active pixels |

The fallback exists because disconnected appearance groups can all span the full
image: the split removes pixels but provides no useful zoom. `spatial_children()`
(`search/partitioning.py:54`) makes four nonempty quadrants and keeps the full
tile extent rather than tightening a fragmented mask.

The split decision, quality metrics, and selected child boxes are saved in
`split.json`; selected crops and an overlay are saved beside it.

## 6. Backup, stopping, and failures

### Reward backup

`_backup_reward()` (`search/tree.py:269`) adds the leaf's raw reward to every node
on the selected path:

```text
visits      += 1
reward_sum  += raw_reward
```

There is no discount and no cost subtraction in the backed-up value. The same
raw reward is used at every level. Cost is only used to rank future branches.

Only **complete** evaluations back up a reward. If a budget interrupts a node
part-way through, the step is recorded with `completed = False` and
`raw_reward = None`, and no node statistics change.

### Stop rules

| Reason | When it fires |
| --- | --- |
| `exhausted` | root has no unevaluated descendants |
| `api_budget` | HTTP call cap reached before the next iteration or after a completed evaluation |
| `local_budget` | the next local generation would exceed `max_local_scores` |
| `iterations` | `max_iterations` reached while work remains |
| `match` | `stop_on_match` is true and a proposal is verified with score >= `match_threshold` |
| `error` | a non-recoverable execution error occurred |

`stop_on_match` defaults to `False`, so the search normally continues until the
budget, iteration, or exhaustion rule stops it.

### Recoverable and fatal failures

- **Invalid verifier JSON** (`InvalidScoreResponse`):
  - record `evaluation_error` on the node and step;
  - expand the node so its children can still produce valid evidence;
  - do not back up an invented zero;
  - continue the search.
- **Fatal execution error** (`RuntimeError` or `ValueError` other than the
  invalid-score case):
  - record the failed step;
  - stop with `error`;
  - keep all earlier evidence.
- **Budget exception**:
  - record the incomplete step;
  - stop with `local_budget` or `api_budget`;
  - never back up the partial evaluation.

## 7. Final ranking and artifacts

`_attach_rankings()` (`search/runner.py:131`) sorts:

- clusters by `score` descending;
- proposals by `(score, sam3_score or 0)` descending.

The first proposal is the **top-1**. The inspection report also records the
**best** IoU among all proposals, which can differ from top-1 if ranking and
ground truth disagree.

`save_search_result()` (`reporting/artifacts.py:28`) writes:

- `results.json`: query, config, usage, stop reason, node summaries, trace,
  clusters, proposals;
- `masks.npz`: node and proposal masks;
- `clusters/`, `proposals/`, `evidence/`, `region_inputs/`: actual crops and
  verifier inputs;
- `preview.jpg`: ranked boxes drawn on the original image.

Ground-truth inspection is a separate post-run step in `tools/run.py`. The
annotations are not passed to the search.

## 8. Cost behavior

The dominant cost is local Qwen image generation. The default `region` mode uses
one generation for up to five candidates on a nonempty region. The `candidate`
mode uses one generation per checked candidate, up to the limit. The resolver
adds one text-only generation at startup.

`state.charge_local()` increments `local_score_calls` before each generation and
raises when the budget is exhausted. A repair attempt after malformed JSON is a
separate generation. `charge_request()` is the parallel budget for a custom
scorer that counts HTTP-style calls; production local scoring uses the local
ledger instead.

Caches reduce repeated work:

- `detection_cache` avoids rerunning SAM3 for the same node;
- `score_cache` reuses an identical crop/query/view assessment;
- `checked_cache` records candidate checks;
- `split_cache` ensures one split per node.

The tree can still revisit a region's branch many times. A node's own evaluation
runs once; later visits usually select its descendants.

## 9. Design invariants

- Inputs and evidence sheets are immutable; mutable run state is explicit.
- By default the verifier receives the full search description, not just the
  detector phrase. `candidate_query_mode` can opt into the detector phrase.
- The detector prompt can be changed by the target resolver, but the original
  query is preserved in every result and artifact.
- Candidate masks and regions are kept in original-image coordinates when
  reported; local coordinates are preserved for audit.
- Every pruning or fallback decision is emitted as an event and saved as an
  artifact when `artifact_dir` is enabled.
- No annotation or ground-truth box is read during inference.

## 10. Limitations

- The reward is a hand-weighted heuristic. A single verified box can produce a
  high score, and the diversity term is maximally favorable for one box.
- Qwen's `Yes/No/Uncertain` judgments are not independently verified and are not
  calibrated.
- SAM3 can miss the target at every scale; the tree can only split regions it
  has, so a detector miss is not recoverable by search alone.
- The target resolver can choose the wrong object when the relational phrasing is
  ambiguous. The fallback keeps the heuristic concept, which may also be wrong.
- Semantic splitting can fall back to quadrants, which may separate an object
  across sibling regions.
- The UCT adaptation does not inherit the fixed-workflow regret guarantee of the
  original Agent-UCT paper.
- IoU against a single annotated box is a localization proxy. It is not answer
  accuracy, and it can reward a box that contains the right object for the wrong
  reason.
