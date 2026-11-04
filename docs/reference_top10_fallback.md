# Reference SAM3 low-score fallback

With `sam3_fallback_top_k=10`, each individual SAM3 call uses the existing
`score > 0.25` filter first. Only when none pass does it retain the 10 highest
native query boxes, sorted by the same sigmoid-logit times sigmoid-presence
score. No second model forward is needed. Scores are never promoted or altered.
Selected boxes are tagged `sam3_selection=top_k_fallback`. They bypass the
ordinary search-layer threshold filter, then go through the existing geometric
validity checks, same-reference NMS, and Qwen verification/answer gate.
Top 10 is before invalid-box removal/NMS, so fewer distinct boxes can remain.
The run's verifier candidate limit is 10 per visited region (combined across
reference prompts); the physical search-generation budget remains 32.
A low-score box is only a candidate. All candidates may be rejected by Qwen.
The default fallback is zero (disabled), preserving previous experiments.

The 15-image selection uses the dataset's existing image-level labels L1, L2,
and L3. Seed 42 samples five question-bearing images per level, all questions
on each image; UNKNOWN is excluded. IDs and labels are the only selection data.
No answers, target/reference geometry or fallback predictions are used. These
are image density strata, not independently calibrated question difficulty.
The selection manifest is frozen before running inference.

Report naming follows the user-confirmed order: L1 = hard, L2 = medium, L3 = easy. Five images per level are sampled with seed 42.
