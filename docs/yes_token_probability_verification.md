# Question-region verification uses Yes/No token probability

The previous question-region verifier generated JSON including a self-reported score. Those historical scores are not token probabilities. Existing output directories remain unchanged and must not be mixed with corrected outputs.

The corrected verifier asks the original question's answerability using the same full-scene overview, full rectangular detail crop and coordinate metadata. It asks for Yes or No only. One forward pass reads next-token logits BEFORE any answer/rationale is generated. The score is softmax([logit_Yes, logit_No])[0], i.e. P(Yes | next token in {Yes, No}), in [0,1]. This is not a full-vocabulary probability nor a calibrated probability of final answer correctness. Full-vocabulary Yes/No probabilities and token IDs/logits are also saved as evidence.

The label is the higher-probability binary token (exact tie: Uncertain). Continuous probability is backed up to search even if the label is No; No no longer forces probability reward to zero. Answer eligibility still requires Yes and score >=0.5. The user-configured early-stop threshold 1.0 remains unchanged; exact 1.0 is ordinarily unattainable except numerical saturation, so runs generally continue to another stopping rule. This is distinct from final-answer eligibility.

One forward-pass score consumes one physical search-generation slot in the existing accounting ledger. There is no generated JSON to parse/repair for verification, so retries no longer consume extra score slots. Object-prompt extraction and final answer generation remain separate generations.

The implementation normalizes to P(Yes|Yes,No); CVSearch's Qwen wrapper additionally maps the same binary probability to 2*p-1. Our retained [0,1] scale matches local search configuration; thresholds cannot be copied between those scales without conversion.

Configuration: algorithm/configs/qwen8_spatial_context_yesprob_budget32_depth5.json. Old candidate-box/legacy scoring modes are outside this correction; only original-question region verification changes. Old audits' generated No/score explanations are historical evidence, not outputs of this new verifier.
