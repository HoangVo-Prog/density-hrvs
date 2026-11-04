# LLMDet primary detector with Qwen locating statements

The prior random20 SAM3 run is stopped with 9/32 questions completed. Retain it unchanged.

New opt-in configuration: `qwen8_llmdet_random20.json`. The primary detector is the official `iSEE-Laboratory/llmdet_tiny`, revision `d05199165a19320a9236396e20fca0a5e065189c`, cached on the secondary drive. Transformers integration uses a direct prompt string and the installed GroundingDinoProcessor postprocessor, with input IDs. No SAM3 model is loaded in this configuration. Grounding DINO fallback and separate owner/part refinement are disabled. LLMDet directly receives the full requested target statement, including the cargo bed if requested.

Qwen 8B rewrites the original question once into `Locate ...`, preserving known attributes, parts, references and spatial constraints. It removes the unknown attribute question without inventing its answer. The generation is counted as a resolver call. Invalid rewrites fail explicitly; they do not silently fall back to a category noun. The original question remains available for final answering. The detector's exact statement is saved in the result configuration.

Reuse K=5 adjacency merging, geometry checks and spatial fallback; use cached full-image DINO features. Qwen uses the same verification and answer views as before. Confidence stays `detector_score` with `sam3_score=null`; rectangular masks are explicitly `box_only`. Existing candidate fusion uses Qwen-only scores for non-SAM3 proposals rather than treating external detector confidence as calibrated SAM3 confidence. Old SAM3-named compatibility fields/events remain in the historical schema.

Validation: 46 tests and 4 subtests passed. Qwen rewrote 012_q0001, 012_q0002 and 015_q0001 as the intended full locating statements. Real CUDA full-image LLMDet probes returned 0, 1 and 1 boxes respectively at threshold .25; the two boxes were broad. These results are plumbing checks, not proof of improved localization or relational reasoning. Full crop-search integration needs separate validation.

Official references:
- https://github.com/iSEE-Laboratory/LLMDet/blob/main/hf_model/README.md
- https://github.com/huggingface/transformers/blob/v5.9.0/docs/source/en/model_doc/mm-grounding-dino.md
