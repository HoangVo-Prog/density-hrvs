# Combined 134-question validation protocol

The full 1,060-question run is stopped. Validate on the existing 134 development questions
before proceeding, as requested by the user. Do not mix its results with the larger export.

Combined arm: pinned 8B NF4, staged CUDA execution, SAM3 owner proposals, explicit owner
identity/spatial verification, linked SAM3 part proposals, one wider part-crop retry if
empty, and bounded Grounding DINO recovery. Maximum two part SAM3 calls and two Grounding
DINO calls per question. Primary search and part verification share 32 physical Qwen search
generations, including repairs; six are reserved for final part verification. Resolver and
one final answer are separately counted. Comparative reranking stays off after the completed
8B 134-pool experiment regressed localization 45 -> 42 (five gains, eight regressions).

Control arm: identical checkpoint, rendering, resolution, total generation cap and runtime,
with structured/part/crop-retry/Grounding DINO switches off. Both arm snapshots must be
prepared from the same source before execution. Run serially on the same GPU.

Report separately: existing owner/proposal top-1 IoU; retained owner-pool coverage;
answer-focus IoU (verified attribute bbox when present, otherwise the selected eligible
owner bbox, exactly as the answerer uses it); exact normalized complete-answer accuracy;
abstentions; joint answer-focus/exact accuracy; actual model calls, elapsed time and memory.
The answer-focus rule is fixed before evaluation and never chooses a box using annotation
IoU. Linked-box coverage is an oracle diagnostic, not top-1 accuracy. Keep entity and part
boxes and their verification provenance in each result.

Completion gates: all 134 cases finish or errors are explicitly resolved/reported; all
physical/detector caps hold; no answer comes from a rejected/unchecked candidate; assess
paired gains and regressions against the matched 8B control, including the 26 original joint
successes; inspect ownership/part failures. A larger-model answer-only gain does not qualify
the combined pipeline by itself. Do not launch the full export merely because code tests pass.


The initial integration exposed `part_concept=color`, which is not a physical object.
The resolver now separates a short detector category from the complete entity description
and rejects properties such as color/pattern/shape as physical parts. Full relations remain
in owner verification. A question-only 134-case 8B GPU preflight produced 134 structurally
valid physical plans, one generation each (`query_plans134_qwen8_v3`). This preflight is not
image accuracy or proof that every semantic role is correct; unknown-object/reference
confusions remain an evaluation concern. The corrected full image runs use the v3 output
folders and fresh immutable snapshots. Earlier interrupted attempts are not merged with them.
