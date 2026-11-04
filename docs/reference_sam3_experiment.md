# Reference-only SAM3 experiment

Opt in with `algorithm/configs/qwen8_reference_random20.json`.
Qwen reads only the original question, extracts a target description and zero or
more reference noun phrases with known colors/attributes. Spatial qualifiers are
kept separately for verification, never concatenated into SAM3 text.
Each visited search region receives one SAM3 call per reference, independently.
All returned boxes retain reference ID and prompt. The existing DINO cached
features, effective adjacent merge, AgentUCT, verification and final answer path
remain. Verifiers check reference identities and locations; the answerer receives
the original question on its usual padded crop around the chosen reference.
No new Qwen box predictor or target detector was added. Part detection and
Grounding DINO fallback are off so SAM3 receives only reference prompts.

Reference boxes are NOT ground-truth target boxes. Legacy target IoU metrics in
the generic benchmark do not measure reference localization and should not be
used to compare this run to target-detection runs. Answer accuracy still needs
separate evaluation. A crop around one reference can omit distant targets or
other references; no success is assumed. If there is no described physical
reference, do not invent a target prompt: there are no SAM3 calls/proposals and
the existing answer gate abstains. Every extracted reference has a separate
bounding-box image under `<question>/references/`, including zero-box cases.
The threshold for this experiment is 0.25, matching the turquoise-car probes.
