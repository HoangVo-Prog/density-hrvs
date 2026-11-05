# algorithm_refactored

A behavior-preserving restructuring of `algorithm/` (DINOv3 → SLIC → K-means →
SAM3 → local Qwen3-VL verifier, driven by cost-aware UCT). `algorithm/` is left
untouched.

## Layout (dependencies point toward `core/`; no import cycles)

```
tools/        CLIs: run, legacy_cli, benchmark, walkthrough, audit/{probes,results,visuals}
reporting/    artifacts (saved results), inspection (ground truth), run_viewer (HTML)
search/       runner      thin facade: run_visual_search()
              runtime     SearchRequest (input) + RunState (per-run caches/budgets)
              setup       request validation + Qwen target resolution + lazy model assembly
              pipeline    evaluate_node / split_node / estimate_node orchestration
              evaluation  verifier/detector primitives
              candidates  whole-region top-5 verification or per-candidate evidence sheets
              tree        explore_tree (UCT selection, backup, stop rules)
              partitioning, region_scoring, config
models/       sam3, qwen_scoring, scoring, regions (DINO/SLIC), evidence sheets
core/         contracts (TypedDicts), errors, events, imaging, geometry, query, data, diagnostics
```

`main.py` keeps the old `algorithm.main` import surface; the legacy CLI lives in
`tools/legacy_cli.py` and imports model-heavy modules only when invoked.

## Docs

- [Algorithm walkthrough](docs/algorithm_walkthrough.md)
- [Pseudocode](docs/pseudocode.md)
- [Success case: 0411_q01](docs/case_study_0411_q01.md)
- [Overleaf documentation template](docs/overleaf_template.tex)
- [Paper-style algorithm walkthrough (LaTeX)](docs/algorithm_walkthrough_section.tex)

## Run

```bash
outputs/cvsearch_runtime/bin/python -u -B -m algorithm_refactored.tools.run --interactive --device cuda
outputs/cvsearch_runtime/bin/python -u -B -m algorithm_refactored.tools.benchmark --help
```

## Library entry point

```python
from algorithm_refactored.search.runner import run_visual_search, save_search_result
from algorithm_refactored.search.config import default_search_config
```

## Design notes

- `TypedDict` contracts at every module boundary keep arrays, model provenance,
  and diagnostics explicit without runtime class overhead.
- `SearchRequest` and `CandidateEvidence` are frozen dataclasses: inputs and
  evidence sheets cannot be mutated after they are built.
- `RunState` owns mutable per-run caches, budgets, and artifact targets; pure
  helpers receive it explicitly instead of closing over hidden state.
- `LocalScorer` is a `Protocol`, and `scorer`, `detector`, and
  `feature_extractor` are injected through `run_visual_search`, so tests and
  benchmarks can replace any model without inheritance.
- Qwen acts as the SAM3 **target resolver**: it receives the full search
  description and chooses the object/region to localize, rather than being
  constrained to the initial noun parsed from the question.
- The default `candidate` verification mode checks each candidate on a zoomed
  context/detail sheet. The experimental `region` mode sends the whole cluster
  with numbered top boxes in one Qwen ranking call and falls back on ties.
- `discovery_mode="sam3_qwen"` optionally runs a tiled SAM3 scan before the tree,
  reranks the candidates with Qwen, and adds the best regions as root seeds while
  keeping the normal semantic/spatial children as a fallback.
- The post-search answer stage crops a padded window around the top-ranked box
  and asks Qwen the original question, writing `final_answer` (answer, bbox,
  crop, prompt, raw response) into `results.json`; disable it with
  `answer_question=False` or `--no-final-answer`.
- `main.py` is a small compatibility facade; CLIs live under `tools/` and load
  model-heavy modules lazily.

## Equivalence

With the wall clock frozen, the refactored runner was diffed against `algorithm.runner`
using deterministic fake models. The diff covered the result dict, every event,
verifier calls and artifact file sets. The scenarios were: default,
`sam3_first=False`, legacy API budget, local budget, invalid verifier replies,
fatal errors, and stop-on-match. All of them were identical.
