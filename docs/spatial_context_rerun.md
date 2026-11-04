# Original-scene context for region verification and answering

This opt-in run changes only Qwen's spatial evidence relative to the corrected
question-region pipeline. Each verification receives the unchanged original
question, two separate images, and deterministic crop-to-scene metadata:
1. Original-scene overview, maximum side 768, with a cyan node-bbox outline.
   For irregular merged regions, yellow shows the active cluster-mask boundary.
2. Clean region crop at its existing resolution, without SAM3/object annotations.
3. Original size, global xyxy region bbox, normalized xyxy bbox, coverage percent,
   image position of region center, and explicit original-image coordinate frame.

No object position, intersection coordinates, answer or target annotation is
injected. A center-location label describes image geometry only. The prompt
requires the overview for positions relative to the intersection or references,
and requires the asked-about target to appear in the clean REGION. Targets
visible only elsewhere in the overview cannot justify accepting a wrong region.

The final answer uses the same overview + selected clean region + mapping, so it
does not lose the context restored at verification. Actual overview, clean crop,
metadata, prompts and raw Qwen responses are saved, including full node stages.
Score-cache identity includes the overview pixels and mapping as well as crop
pixels and question, preventing equal-looking crops at different scene positions
from sharing an assessment.

SAM3 requested-object/reference prompts, independent per-prompt >0.25 filtering
and own top-10 fallback, DINO cache, SLIC/K-means adjacent merge, K=5, AgentUCT,
32-search-generation budget and the exact 15-image/32-question selection remain.
Levels: L1 easy, L2 medium, L3 hard. This is inspired by the context-preservation
motivation of RAP, not an implementation of its compressed patch-layout method.
