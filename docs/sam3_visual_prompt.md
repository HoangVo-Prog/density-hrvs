# Simplified spatial prompts and SAM3 boxes

All batch runs were cancelled at the user's request. No new model inference is part of this change.

Spatial verification and final answering now show SAM3 detections on BOTH the overview and detail crop, with matching IDs. T groups are target prompts (magenta), R groups are reference prompts (orange); cyan still identifies the current region. Individual candidates use IDs such as T1.1 and R1.2. Detection-local xyxy coordinates map through the node offset and overview scale; invalid/empty boxes are skipped. Retained proposals are not capped or re-filtered by this display step. GT annotations are never read by it.

The prompt states that boxes are suggestions and annotation colors are not object attributes. Left/right/above/below always mean displayed-image directions, never the object's own facing direction. It retains the original question, image size, global region bbox and short prompt-group legend, removing verbose coordinate prose and repeated instructions.

Verification still reads P(Yes|Yes,No) from next-token logits. No generated self-rating or JSON rationale. The same box rendering is used when preparing final-answer inputs. Full box geometry and detector confidence remain in metadata for audit, while the prompt contains only a compact legend.

032_q0002 preview was rendered from previously saved SAM3 detections, without rerunning SAM3 or Qwen. Old outputs remain historical records. Tests cover projection/clipping, source pixel preservation, role IDs and actual routing into both verification and answering.
