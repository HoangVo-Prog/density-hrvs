# Full dataset rerun with search-generation budget 32

User requested the full dataset after the deeper first-question search succeeded.
Use the same pipeline/configuration as that successful replay, changing only the
per-question physical search-generation budget from 50 to 32. The older full
run used budget 10 and depth 3 and remains stopped.

New run: max_local_scores=32, max_iterations=50, max_depth=5,
stop_on_match=true, match_threshold=1.0. Stop early at a verified Yes 100/100.
JSON-repair retries count toward the 32 search generations. Resolver and final
answer generations are counted separately. At most 50 node visits are allowed;
the generation budget usually stops search sooner. Created, unvisited child
nodes may make the displayed tree larger than the number of actual visits.

Include all 500 question-bearing images and all 1,060 questions from the official
export. Image 223 has no question. No difficulty filtering or prior prediction
reuse. Fresh frozen source/configuration and a separate output directory.

Unchanged: original question plus overview/clean crop/global-position mapping
for Qwen verification and answering; target plus reference prompts for SAM3;
independent per-prompt >0.25 selection or own top-ten fallback with raw confidence
preserved; DINO full-image feature cache; K=5 effective adjacent merging; AgentUCT;
full node/K-means stage exports.

Reporting-only repair: missing target-box annotation (e.g. 039_q0001) no longer turns a saved inference result into a TypeError. Inspection marks IoU unavailable and preserves the model answer. No model input, search behavior or prediction is changed by this repair.
