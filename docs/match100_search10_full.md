# Full-dataset spatial-context run: match 1.0, search 10

User-requested settings: stop_on_match=true, match_threshold=1.0,
max_iterations=10, max_local_scores=10. A verified Yes at Qwen's normalized
score 1.0 stops early. Scores below 1.0 or unverified ratings do not. Otherwise
stop after at most 10 attempted node visits or 10 physical search generations
(including JSON-repair retries), whichever comes first. Resolver and final-answer
generations are counted separately. The tree can create more than 10 child nodes;
the visit/evaluation budget is 10, not a cap on generated tree metadata.

All 500 question-bearing images and all 1,060 questions from the official export
are included. Image 223 has no questions and is recorded as excluded. No image
level filtering. New output directory, without mixing prior predictions.

Unchanged: overview + clean crop + exact coordinate metadata for verification
and answering, requested-object/reference SAM3 prompts, independent per-prompt
>0.25 or own top-10 fallback, DINO feature cache, K=5 effective adjacent merging,
AgentUCT, maximum depth 3, and full node/K-means artifact export.
