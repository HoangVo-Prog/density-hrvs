# Brief reason-first JSON for verification and final answering

Config reasoned_json=true enables this optional output mode. The original short two-view prompt still measures next-token P(Yes|Yes,No) in one forward pass. A SEPARATE model generation then gives a short visible-evidence explanation of that probability-based label, returning reason before answer. Its label is validated against the binary probability decision. The explanation is post-hoc and does not affect the probability. Both prompts and raw explanation JSON are saved; no textual numeric rating is used.

This separation preserves the user's token-probability requirement. The verifier reason is not evidence that the model reasoned before computing the probability; only its JSON presentation and explanation generation are reason-first. The final answer generation itself emits a reason before its concise answer, which are both saved separately plus ordered response_json. Neither generation sees GT.

Each verifier explanation and any JSON-repair retry consumes an additional physical search-generation slot, while the final answer and its repair belong to the separate answer stage. Budget32 therefore includes probability forwards plus verifier explanations. The rerun keeps early-stop threshold0.95, same object prompts, refreshed SAM3 boxes, spatial mapping and image-view directions. No full dataset run is launched.
