# Requested detect–verify–split flow

Question -> Qwen object plan (once). Only requested physical target/parts and physical reference objects become SAM3 prompts; scene positions remain context.

Node0 full scene -> each extracted prompt gets its own SAM3 call -> save per-prompt raw/confidence-filtered boxes -> draw matching target/reference IDs on Qwen inputs -> verify the original question with next-token P(Yes|Yes,No). If the configured match condition succeeds, answer without partitioning. Otherwise partition node0.

For each selected child: run each same object prompt on the closer rectangular crop, save fresh local detections and mapped global boxes, draw those boxes on full overview and detail, verify original question, and split this child only if search continues. Crop detections never reuse the root detection set; detection caching is by node ID. A closer crop may improve confidence, but is not guaranteed to do so.

The previous initialization split node0 without evaluating it. Question-region mode now evaluates root first. Other verification modes retain previous initialization. Early stopping still uses the configured probability threshold (currently1.0); this change does not silently alter it.

All experiment runs remain stopped; no new GPU model inference was run for these changes. Object-extraction, per-prompt routing, coordinate projection, actual verifier/answer inputs and root-before-split order have regression tests.
