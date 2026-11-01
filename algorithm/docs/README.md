# algorithm_refactored documentation

This directory documents the refactored bounded recursive visual search in
`algorithm_refactored/`. Start with the walkthrough, then use the pseudocode as a
compact implementation map. The case study applies both to one saved run.

- [Algorithm walkthrough](algorithm_walkthrough.md) - data flow, cost-aware UCT,
  region evaluation, reward construction, splitting, and stop rules.
- [Pseudocode](pseudocode.md) - simplified control flow for the main loop and
  each stage.
- [Success case: 0411_q01](case_study_0411_q01.md) - a saved run from
  `outputs/official_part1_resolver/` with top-1 IoU 0.957.
- [Overleaf template](overleaf_template.tex) - a paper-style LaTeX template for
  input/output, notation, a pipeline diagram, pseudocode, math, and a worked
  example. See the [template notes](overleaf_template_README.md).
- [Paper-style algorithm walkthrough](algorithm_walkthrough_section.tex) - a
  section-only LaTeX write-up with search-tree notation, region reward,
  cost-aware UCT, backup, splitting, stopping rules, and pseudocode.
- [Run and I/O formats](run_and_io.md) - prerequisites, CLI examples, input
  formats, `results.json` schema, benchmark output, and the final-answer boundary.

## What the algorithm does

The system receives an RGB image and a search description. It returns ranked
image regions (SAM3 boxes), the search tree, an allocation trace, usage counters,
and a stop reason. A final stage then answers the original question from a crop
around the top-ranked region (`final_answer`). The search itself is a
**localization search**; the text answer is produced by one separate Qwen
generation after the search stops.

The search alternates four operations:

1. **Select** one frontier leaf with cost-aware UCT.
2. **Evaluate** that rectangular region: SAM3 proposes boxes, and Qwen3-VL
   verifies the candidates against the search description.
3. **Back up** the region's raw reward through the selected path.
4. **Split** the region into smaller semantic groups, re-encoding the crop with
   DINOv3, SLIC, and K-means. A spatial quadrant fallback keeps the tree finite
   when semantic geometry is poor.

```text
image + question
  -> question_target() -> search description
  -> prepare_run() -> config, models, caches, budgets
  -> root = full image
  -> repeat until a stop rule:
       UCT selects a frontier leaf
       evaluate leaf: SAM3 -> NMS -> Qwen verification -> region reward
       back up reward along the path
       split leaf: DINOv3 -> SLIC -> K-means, or spatial quadrants
  -> rank all observed candidates
  -> answer original question from top-1 region crop
  -> results.json + final_answer_region.png + stage artifacts
```

## Module map

| Path | Responsibility |
| --- | --- |
| `tools/run.py` | CLI and dataset question loading; calls the search |
| `search/runner.py` | Thin orchestration: prepare, explore, rank, save |
| `search/setup.py` | Request validation, target resolver, lazy model loading |
| `search/runtime.py` | `SearchRequest`, mutable `RunState`, caches, budgets |
| `search/tree.py` | Cost-aware UCT, backup, expansion, stop rules |
| `search/pipeline.py` | Node evaluation, split orchestration, cost estimate |
| `search/evaluation.py` | SAM3 detection and score-cache primitives |
| `search/candidates.py` | Whole-region and per-candidate Qwen verification |
| `search/region_scoring.py` | NMS, candidate fusion, region reward components |
| `search/partitioning.py` | DINO/SLIC/K-means children and spatial fallback |
| `models/regions.py` | DINOv3 PCA, SLIC entropy selection, K-means |
| `models/sam3.py` | SAM3 loading, proposal selection, mask postprocessing |
| `models/qwen_scoring.py` | Target resolver and Qwen verifier prompts |
| `models/evidence.py` | Region sheets and context/detail evidence sheets |
| `reporting/artifacts.py` | `results.json`, masks, crops, previews |
| `reporting/inspection.py` | Post-run ground-truth overlays |

## Reading a saved run

The case study uses this directory:

```text
outputs/official_part1_resolver/0411_q01/
  completion.json          compact outcome (metrics, calls, errors)
  run/results.json         tree, trace, usage, ranked clusters and proposals
  run/proposals/           raw SAM3 crops
  run/evidence/            exact Qwen verifier inputs
  run/masks.npz            node and proposal masks
  stages/tree/             chronological tree snapshots and allocation table
  stages/nodes/node_NNNN/  DINO, SLIC, K-means, split, detection, verification
  inspection/              post-run ground-truth overlays (not model input)
```

Open `stages/tree/index.html` to see the tree and iteration table. Open a
`stages/nodes/node_NNNN/index.html` page to see that region's actual model
inputs and stage artifacts. The `inspection/` directory is created after
inference and is never read by the search.

## Scope and measurement

The walkthrough describes the refactored implementation, including the default
SAM3-first path. The legacy whole-region path remains available with
`sam3_first=False` and is mentioned only where it changes behavior.

The case study is one development example, not a benchmark result. Success is
measured by box IoU against an annotated target box, which is a localization
proxy and not answer correctness. The verifier ratings are heuristic visual
judgments, not calibrated probabilities.
