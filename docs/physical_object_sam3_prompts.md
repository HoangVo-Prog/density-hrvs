# Physical objects versus scene context for SAM3

Orange reference boxes are actual SAM3 outputs, not clustering regions. In the saved 032_q0002 node2 detections, blue bus produces1 box, white-cab pickup3, target cargo bed3, and the invalid scene prompt right-hand approach produces10 top-k fallback boxes. The large orange rectangles came from this last incorrect object extraction.

The object-extraction instruction now explicitly keeps approaches, queues and image sides in context. A narrow parser guard removes recognized scene-position phrases from reference objects before separate SAM3 calls, records them in context_only, and rejects a scene position as the target. It does not implement a complete physical-object ontology. Valid target/reference detections still use their own >0.25/top10 policy and remain individually drawn on both VLM images. Orange=reference, magenta=target, cyan=current region. No extra boxes are invented from K-means for SAM3 overlays.

Corrected preview uses the seven previously saved SAM3 detections from the three physical prompts. This is an offline replay/filter of existing detections, not a new SAM3/Qwen run. All experiment runs stay stopped.
