# Correction: independent object proposals and question-region verification

The reference-only verifier was wrong for the requested algorithm. It checked
whether boxes matched references and could reject a correct region because the
reference itself was not the asked-about object. That behavior is not used here.

1. Qwen reads the original question and extracts a target object/part AND all
   references as separate short SAM3 prompts with explicit roles. Unknown answers
   are not supplied. A target prompt is mandatory even with zero references.
2. Each visited region receives one SAM3 call per extracted object, including
   the requested object. Threshold 0.25 and empty-set top-10 fallback are retained.
   Actual confidence and target/reference role are saved for every box.
3. Verification is a separate Qwen call: ORIGINAL QUESTION + CLEAN WHOLE REGION.
   No reference-only query, candidate crop, colored box, SAM3 confidence, or
   extracted-object list is sent to this verifier. It evaluates answerability
   from visible target, attributes, spatial and reference evidence in the region.
   It runs even when SAM3 returns zero valid boxes.
4. AgentUCT reward is that region's Qwen rating (zero for No). It is not a
   reference-box score or a mixture with SAM3 confidence. Answer eligibility still
   requires a verified Yes. A region score is never transferred to an object box.
5. Final Qwen answer uses the original question and the selected clean REGION,
   with zero additional answer padding. Outputs identify an answer_region rather
   than presenting a reference box as the requested target.

DINO cached full-image features, SLIC/K-means, effective adjacent merge, branching
5, depth/iteration limits and 32 search-generation budget are retained. Top-10
fallback is before NMS. Full per-node/K-means stage exports are enabled through
--save-stages. Selection is the same 15 images, five L1-easy/L2-medium/L3-hard;
all 32 questions are included. It is a new frozen run, never overwritten results.

Whole-region answerability is not precise target localization. Generic target
box IoU for answer regions does not measure answer correctness. The report must
show generated answer, exact verifier question/input, region ID and raw scores.

Each object prompt is thresholded independently. Keep all native boxes above 0.25 (proposal cap disabled in this config); only that prompt's empty thresholded set triggers its own top-10 fallback. NMS is applied within each object ID, not across different prompts.
