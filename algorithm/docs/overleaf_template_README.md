# Overleaf documentation template

Use [`overleaf_template.tex`](overleaf_template.tex) as the starting point.

## Import into Overleaf

1. Create a new Overleaf project.
2. Upload `overleaf_template.tex` or paste it into `main.tex`.
3. Compile with **pdfLaTeX**. The template uses standard TeX Live packages.
4. Replace the gray placeholders and the example case with your own content.

## Template structure

| Section | Purpose |
| --- | --- |
| Problem definition | Input/output tables, scope, non-goals, evidence boundary |
| Notation | One symbol table used consistently everywhere |
| Algorithm at a glance | TikZ pipeline diagram and step-by-step operation table |
| High-level pseudocode | One bounded main loop, implementation details hidden |
| Mathematical formulation | Search state, selection rule, reward, stopping rule |
| Worked example | One real run, a compact results table, and a tree snapshot |
| Operations and cost model | What each stage costs and what is cached/bounded |
| Failure modes | Detector misses, verifier errors, reward bias, truncation |
| Reproducibility checklist | Model IDs, config, raw replies, artifacts, hashes |

## Writing rules

- Put **all symbols** in the notation table and reuse them exactly.
- Put the **main loop** in pseudocode; put model-specific detail in helper
  paragraphs or equations.
- Put each formula **once**. Refer to its equation number from the pseudocode
  and the step table.
- Separate **input**, **model output**, **diagnostics**, and **post-run audit**
  in both the text and the figures.
- In the worked example, state whether the case was selected before or after the
  run. Do not present a selected success case as an unbiased benchmark result.
- State the metric boundary explicitly: a localization IoU is not answer
  accuracy, and a verifier score is not a calibrated probability.
