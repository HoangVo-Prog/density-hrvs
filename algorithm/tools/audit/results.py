"""Read-only failure attribution from saved predictions, stage files and trees.

Ground truth is used exclusively for this post-run audit. Oracle and alternative
ranking results describe the existing candidate pool, not new inference accuracy.
"""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
from statistics import median
from typing import TypedDict

from ...core.geometry import box_iou


class NodeAudit(TypedDict):
    """One visited region, with annotation-derived diagnostics added after inference."""

    node_id: int
    depth: int
    bbox: list[int]
    region_fraction: float
    target_coverage: float  # Fraction of annotated target area inside this crop.
    sam_target_short_side: (
        float  # Approximate target pixels after SAM3's 1008-square resize.
    )
    qwen_score: float
    combined_score: float
    raw_boxes: int
    unique_boxes: int
    mean_confidence: float
    count_score: float
    diversity_score: float
    raw_best_iou: float
    capped: bool
    qwen_reason: str


class QuestionAudit(TypedDict):
    """Exhaustive question-level audit, preserving failures rather than dropping rows."""

    question_id: str
    image_id: str
    question: str
    annotated_answer: str
    category: str
    potential_anchor_attribute_mismatch: bool
    target_bbox: list[float]
    target_area_fraction: float
    concept: str
    status: str
    error: str
    diagnosis: str
    top1_iou: float
    best_iou: float
    raw_best_iou: float
    proposal_count: int
    ranking_policies: dict[str, float]
    winner_node: int
    winner_has_correct_box: bool
    best_box_qwen_score: float
    best_box_sam_confidence: float
    best_box_node: int
    region_oracle_top_sam_iou: float
    root_children_target_coverage: float
    max_visited_target_coverage: float
    first_hit_iteration: int
    nodes_evaluated: int
    target_nodes_evaluated: int
    target_nodes_capped: int
    high_score_wrong_nodes: int
    low_score_target_nodes: int
    best_evaluated_target_scale: float
    best_unvisited_target_scale: float
    max_depth: int
    frontier: int
    fallback_splits: int
    splits: int
    repeated_same_region: int
    nodes: list[NodeAudit]


def coverage(region: list[int], target: list[float]) -> float:
    x1, y1 = max(region[0], target[0]), max(region[1], target[1])
    x2, y2 = min(region[2], target[2]), min(region[3], target[3])
    area = (target[2] - target[0]) * (target[3] - target[1])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / area


def projected_short_side(region: list[int], target: list[float]) -> float:
    """Approximate SAM target scale, meaningful only when target coverage is high."""
    return min(
        (target[2] - target[0]) * 1008 / (region[2] - region[0]),
        (target[3] - target[1]) * 1008 / (region[3] - region[1]),
    )


def audit(root: Path, data_dir: Path) -> tuple[list[QuestionAudit], dict]:
    annotations = {
        r["question_id"]: r
        for r in json.loads(
            sorted(data_dir.glob("hrvs_bbox_samples_*.json"))[-1].read_text()
        )
    }
    manifest = json.loads(
        sorted(data_dir.glob("hrvs_benchmark_export_*.json"))[-1].read_text()
    )
    sizes = {
        r["image_id"]: (
            r["original_resolution"]["width"],
            r["original_resolution"]["height"],
        )
        for r in manifest["images"]
    }
    outcomes = {
        r["question_id"]: r
        for r in json.loads((root / "summary.json").read_text())["outcomes"]
    }
    config = json.loads((root / "config.json").read_text())
    rows: list[QuestionAudit] = []
    node_stats: list[NodeAudit] = []
    for selected in json.loads((root / "selection.json").read_text()):
        qid = selected["question_id"]
        sample = annotations[qid]
        b = sample["target_bbox"]
        target = [
            float(b["x"]),
            float(b["y"]),
            float(b["x"] + b["width"]),
            float(b["y"] + b["height"]),
        ]
        target_tuple = (target[0], target[1], target[2], target[3])
        folder = root / qid
        result_path = folder / "run/results.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        tree_path = folder / "stages/tree/latest.json"
        tree = json.loads(tree_path.read_text()) if tree_path.exists() else {}
        nodes = result.get("nodes", tree.get("nodes", []))
        trace = result.get("trace", tree.get("trace", []))
        proposals = result.get("proposals", [])
        width, height = sizes[selected["image_id"]]
        total_area = width * height
        evaluated: list[NodeAudit] = []
        for node in nodes:
            stage = folder / f"stages/nodes/node_{node['node_id']:04d}"
            score_path = stage / "region_score.json"
            if not score_path.exists():
                continue
            score = json.loads(score_path.read_text())
            component_path = stage / "region_score_components.json"
            components = (
                json.loads(component_path.read_text())
                if component_path.exists()
                else {}
            )
            detection_path = stage / "detections.json"
            detections = (
                json.loads(detection_path.read_text())
                if detection_path.exists()
                else []
            )
            x, y, x2, y2 = node["bbox"]
            raw_iou = max(
                (
                    box_iou(
                        (
                            x + d["bbox"][0],
                            y + d["bbox"][1],
                            x + d["bbox"][2],
                            y + d["bbox"][3],
                        ),
                        target_tuple,
                    )
                    for d in detections
                ),
                default=0.0,
            )
            item: NodeAudit = {
                "node_id": node["node_id"],
                "depth": node["depth"],
                "bbox": node["bbox"],
                "region_fraction": (x2 - x) * (y2 - y) / total_area,
                "target_coverage": coverage(node["bbox"], target),
                "sam_target_short_side": projected_short_side(node["bbox"], target),
                "qwen_score": score["score"],
                "combined_score": components.get("combined_score", 0.0),
                "raw_boxes": len(detections),
                "unique_boxes": components.get("unique_box_count", 0),
                "mean_confidence": components.get("mean_confidence", 0.0),
                "count_score": components.get("count_score", 0.0),
                "diversity_score": components.get("diversity_score", 0.0),
                "raw_best_iou": raw_iou,
                "capped": bool(
                    config["max_proposals_per_node"]
                    and len(detections) >= config["max_proposals_per_node"]
                ),
                "qwen_reason": score["reason"],
            }
            evaluated.append(item)
        ious = [box_iou(tuple(p["bbox"]), target_tuple) for p in proposals]
        best_index = max(range(len(ious)), key=lambda i: ious[i]) if ious else -1
        top1, best = (ious[0] if ious else 0.0), max(ious, default=0.0)
        good_nodes = {
            p["cluster_id"] for p, value in zip(proposals, ious) if value >= 0.5
        }
        winner = proposals[0]["cluster_id"] if proposals else -1
        policies: dict[str, float] = {}
        keys = {
            "sam_confidence_only": lambda p: p["sam3_score"] or 0,
            "candidate_then_confidence": lambda p: (
                p.get("candidate_score", p["verifier"]["score"]),
                p["sam3_score"] or 0,
            ),
            "candidate_times_confidence": lambda p: (
                p.get("candidate_score", p["verifier"]["score"])
                * (p["sam3_score"] or 0)
            ),
            "region_reward_times_confidence": lambda p: (
                p["score"] * (p["sam3_score"] or 0)
            ),
        }
        for name, key in keys.items():
            chosen = max(proposals, key=key) if proposals else None
            policies[name] = (
                box_iou(tuple(chosen["bbox"]), target_tuple) if chosen else 0.0
            )
        per_region: dict[int, dict] = {}
        for proposal in proposals:
            nid = proposal["cluster_id"]
            if nid not in per_region or (proposal["sam3_score"] or 0) > (
                per_region[nid]["sam3_score"] or 0
            ):
                per_region[nid] = proposal
        region_oracle = max(
            (box_iou(tuple(p["bbox"]), target_tuple) for p in per_region.values()),
            default=0.0,
        )
        outcome = outcomes[qid]
        raw_best = max((n["raw_best_iou"] for n in evaluated), default=0.0)
        diagnosis = (
            "execution_error"
            if outcome["status"] != "complete"
            else "top1_hit"
            if top1 >= 0.5
            else "within_winning_region_ranking"
            if winner in good_nodes
            else "region_ranking"
            if best >= 0.5
            else "lost_after_detection"
            if raw_best >= 0.5
            else "no_matching_retained_detection"
        )
        best_proposal = proposals[best_index] if best_index >= 0 else None
        target_nodes = [n for n in evaluated if n["target_coverage"] >= 0.9]
        unvisited = [
            n
            for n in nodes
            if not n["evaluated"] and coverage(n["bbox"], target) >= 0.9
        ]
        split_files = list((folder / "stages/nodes").glob("*/split.json"))
        split_decisions = [json.loads(p.read_text()) for p in split_files]
        boxes = [tuple(n["bbox"]) for n in evaluated]
        concept = result.get("config", {}).get("sam_prompt", "")
        possible_part_prompts = {
            "hat",
            "cap",
            "helmet",
            "wide-brimmed hat",
            "knit hat",
            "shoes",
            "trousers",
            "jeans",
            "glove",
            "shopping bag",
            "shoulder bag",
            "backpack",
            "scarf",
        }
        row: QuestionAudit = {
            "question_id": qid,
            "image_id": selected["image_id"],
            "question": sample["question"],
            "annotated_answer": sample["answer"],
            "category": b.get("category", ""),
            "potential_anchor_attribute_mismatch": b.get("category") == "upper_body"
            and concept in possible_part_prompts,
            "target_bbox": target,
            "target_area_fraction": (target[2] - target[0])
            * (target[3] - target[1])
            / total_area,
            "concept": concept,
            "status": outcome["status"],
            "error": outcome["error"],
            "diagnosis": diagnosis,
            "top1_iou": top1,
            "best_iou": best,
            "raw_best_iou": raw_best,
            "proposal_count": len(proposals),
            "ranking_policies": policies,
            "winner_node": winner,
            "winner_has_correct_box": winner in good_nodes,
            "best_box_qwen_score": best_proposal["verifier"]["score"]
            if best_proposal
            else 0.0,
            "best_box_sam_confidence": (best_proposal["sam3_score"] or 0.0)
            if best_proposal
            else 0.0,
            "best_box_node": best_proposal["cluster_id"] if best_proposal else -1,
            "region_oracle_top_sam_iou": region_oracle,
            "root_children_target_coverage": max(
                (coverage(n["bbox"], target) for n in nodes if n["parent_id"] == 0),
                default=0.0,
            ),
            "max_visited_target_coverage": max(
                (n["target_coverage"] for n in evaluated), default=0.0
            ),
            "first_hit_iteration": min(
                (s["iteration"] for s in trace if s["node_id"] in good_nodes),
                default=-1,
            ),
            "nodes_evaluated": len(evaluated),
            "target_nodes_evaluated": len(target_nodes),
            "target_nodes_capped": sum(n["capped"] for n in target_nodes),
            "high_score_wrong_nodes": sum(
                n["target_coverage"] == 0 and n["qwen_score"] >= 0.8 for n in evaluated
            ),
            "low_score_target_nodes": sum(
                n["qwen_score"] <= 0.2 and n["sam_target_short_side"] >= 32
                for n in target_nodes
            ),
            "best_evaluated_target_scale": max(
                (n["sam_target_short_side"] for n in target_nodes), default=0.0
            ),
            "best_unvisited_target_scale": max(
                (projected_short_side(n["bbox"], target) for n in unvisited),
                default=0.0,
            ),
            "max_depth": max((n["depth"] for n in evaluated), default=0),
            "frontier": sum(not n["evaluated"] and not n["children"] for n in nodes),
            "fallback_splits": sum(s["used_spatial_fallback"] for s in split_decisions),
            "splits": len(split_decisions),
            "repeated_same_region": len(boxes) - len(set(boxes)),
            "nodes": evaluated,
        }
        rows.append(row)
        node_stats.extend(evaluated)
    traces = [
        s
        for path in root.glob("*/run/results.json")
        for s in json.loads(path.read_text())["trace"]
    ]
    alignment_flags = [r for r in rows if r["potential_anchor_attribute_mismatch"]]
    summary = {
        "questions": len(rows),
        "completed": sum(r["status"] == "complete" for r in rows),
        "potential_anchor_attribute_alignment_flags": len(alignment_flags),
        "alignment_flag_top1_hits": sum(r["top1_iou"] >= 0.5 for r in alignment_flags),
        "alignment_flag_candidate_hits": sum(
            r["best_iou"] >= 0.5 for r in alignment_flags
        ),
        "partial_execution_error_candidate_hits": sum(
            r["status"] != "complete" and r["best_iou"] >= 0.5 for r in rows
        ),
        "diagnoses": dict(Counter(r["diagnosis"] for r in rows)),
        "top1_hits": sum(r["top1_iou"] >= 0.5 for r in rows),
        "candidate_hits": sum(r["best_iou"] >= 0.5 for r in rows),
        "raw_detection_hits": sum(r["raw_best_iou"] >= 0.5 for r in rows),
        "oracle_region_selection_with_confidence_box_ranking_hits": sum(
            r["region_oracle_top_sam_iou"] >= 0.5 for r in rows
        ),
        "fixed_pool_ranking_hits": {
            name: sum(r["ranking_policies"][name] >= 0.5 for r in rows)
            for name in next(iter(rows))["ranking_policies"]
        },
        "evaluated_nodes": len(node_stats),
        "high_score_no_target_nodes": sum(
            n["target_coverage"] == 0 and n["qwen_score"] >= 0.8 for n in node_stats
        ),
        "qwen_zero_score_nodes": sum(n["qwen_score"] == 0 for n in node_stats),
        "zero_qwen_but_reward_above_0_2_nodes": sum(
            n["qwen_score"] == 0 and n["combined_score"] > 0.2 for n in node_stats
        ),
        "count_reward_saturated_nodes": sum(n["count_score"] >= 1 for n in node_stats),
        "median_diversity_score": median(n["diversity_score"] for n in node_stats),
        "median_root_exploration_bonus": median(t["exploration_bonus"] for t in traces),
        "median_selected_root_quality": median(t["quality"] for t in traces),
        "median_root_cost_penalty": median(
            t["estimated_remaining_seconds"]
            * config["cost_weight"]
            / config["reference_cost_seconds"]
            for t in traces
        ),
        "questions_with_high_score_no_target": sum(
            r["high_score_wrong_nodes"] > 0 for r in rows
        ),
        "target_containing_nodes": sum(n["target_coverage"] >= 0.9 for n in node_stats),
        "target_containing_nodes_qwen_at_most_0_2": sum(
            n["target_coverage"] >= 0.9 and n["qwen_score"] <= 0.2 for n in node_stats
        ),
        "target_containing_capped_nodes": sum(
            n["target_coverage"] >= 0.9 and n["capped"] for n in node_stats
        ),
        "fallback_splits": sum(r["fallback_splits"] for r in rows),
        "splits": sum(r["splits"] for r in rows),
        "median_frontier": median(r["frontier"] for r in rows),
        "questions_root_children_cut_target": sum(
            r["root_children_target_coverage"] < 0.9 for r in rows
        ),
        "questions_no_visited_crop_retains_90pct_target": sum(
            r["max_visited_target_coverage"] < 0.9 for r in rows
        ),
        "misses_with_1_5x_better_unvisited_target_scale": sum(
            r["best_iou"] < 0.5
            and r["best_unvisited_target_scale"]
            > 1.5 * r["best_evaluated_target_scale"]
            for r in rows
        ),
        "retained_detection_misses_at_32px_target_scale": sum(
            r["status"] == "complete"
            and r["raw_best_iou"] < 0.5
            and r["best_evaluated_target_scale"] >= 32
            for r in rows
        ),
        "notes": [
            "Ground truth used only after inference.",
            "High-score/no-target counts use the annotated target's location; annotation ambiguity remains possible.",
            "Target scale uses SAM3's 1008-square resize, not Qwen's processor.",
            "Alternative rankings reuse the fixed existing pool; these are not independently validated improvements.",
            "Retained detections were capped at 16; a miss does not prove SAM3 never produced a lower-confidence matching box.",
        ],
    }
    return rows, summary


def write_report(root: Path, rows: list[QuestionAudit], summary: dict) -> Path:
    output = root / "audit"
    output.mkdir(exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps({"summary": summary, "questions": rows}, indent=2)
    )
    table = ""
    for row in rows:
        qid = row["question_id"]
        inspection = (
            f'<a href="../{qid}/inspection/index.html">Inspection</a>'
            if (root / qid / "inspection/index.html").exists()
            else "No exported report"
        )
        alignment = (
            " [anchor/attribute review]"
            if row["potential_anchor_attribute_mismatch"]
            else ""
        )
        table += f'<tr><td>{html.escape(qid)}</td><td>{html.escape(row["diagnosis"] + alignment)}</td><td>{row["top1_iou"]:.3f}</td><td>{row["best_iou"]:.3f}</td><td>{row["raw_best_iou"]:.3f}</td><td>{row["nodes_evaluated"]}</td><td>{row["best_evaluated_target_scale"]:.1f}</td><td>{row["high_score_wrong_nodes"]}</td><td>{row["target_nodes_capped"]}/{row["target_nodes_evaluated"]}</td><td><a href="../{qid}/stages/tree/index.html">Tree</a> {inspection} <a href="visuals/{qid}.jpg">Review card</a></td></tr>'
    (output / "index.html").write_text(
        f'<html><meta charset="utf-8"><style>body{{font:16px system-ui;margin:25px}}td,th{{padding:7px;border:1px solid #ccc}}table{{border-collapse:collapse}}</style><h1>All-question failure audit</h1><p>Read-only analysis of saved results. Exact counts include all execution failures. Model blame requires controlled reruns; coordinate/score diagnostics identify candidates for visual review.</p><pre>{html.escape(json.dumps(summary, indent=2))}</pre><table><tr><th>Question</th><th>Diagnosis</th><th>Top1 IoU</th><th>Best IoU</th><th>Raw IoU</th><th>Evaluated regions</th><th>Best target px at SAM input</th><th>High score but no target</th><th>Capped/target regions</th><th>Saved evidence</th></tr>{table}</table></html>'
    )
    write_findings(output, summary)
    return output


def write_findings(output: Path, summary: dict) -> None:
    """Tie recommendations to original traces and separate diagnostic reruns."""
    grounding_path = output / "probes/grounding_probes.json"
    detector_path = output / "probes/detector_probes.json"
    grounding = (
        json.loads(grounding_path.read_text()) if grounding_path.exists() else []
    )
    detector = json.loads(detector_path.read_text()) if detector_path.exists() else []
    original_winners = [
        r for r in grounding if r["region_selection"] == "original_winning_region"
    ]
    probe_summary = {
        "original_winning_region_cases": len(
            {r["question_id"] for r in original_winners}
        ),
        "numbered_region_hits": sum(
            r["mode"] == "numbered_region_only" and r["iou"] >= 0.5
            for r in original_winners
        ),
        "overview_plus_region_hits": sum(
            r["mode"] == "overview_plus_numbered_region" and r["iou"] >= 0.5
            for r in original_winners
        ),
        "total_qwen_probe_calls": len(grounding),
        "sam_probe_variants": len(detector),
        "sam_probe_questions": len({r["question_id"] for r in detector}),
        "sam_cap_only_recovered_hits": sum(
            r["best_iou_above_0_1"] >= 0.5 and r["best_iou_top16_above_0_1"] < 0.5
            for r in detector
        ),
        "disclaimer": "Selected failure cases; SAM crops chosen with ground-truth visibility after the run. CPU float32 diagnostic reruns differ from production GPU execution. No generalization or end-to-end accuracy claim.",
    }
    (output / "probe_summary.json").write_text(json.dumps(probe_summary, indent=2))
    sections = [
        (
            "Conclusion",
            "The result mixes target-definition errors, final box-selection errors, actual VLM grounding mistakes, filtered SAM3 proposals and execution failures. The saved evidence does not support blaming SAM3 alone. Original production predictions and their reported accuracy remain unchanged by this audit.",
        ),
        (
            "What was inspected",
            f"All {summary['questions']} question traces and {summary['evaluated_nodes']} evaluated-region records were audited. Review cards were generated for every question, showing the full image, actual winning crop, annotated-target context and detail. Representative failures were visually inspected. Controlled probes made {len(grounding)} local Qwen calls and {len(detector)} SAM3 calls across {probe_summary['sam_probe_questions']} detector cases. Ground truth was used only after the production run, including to select diagnostic crops.",
        ),
        (
            "Observed outcomes",
            "27 correct top-ranked boxes; 21 completed runs with a correct box in the winning region but a different box selected; 18 completed runs selecting a different region despite a correct candidate elsewhere; 58 completed runs with no matching retained box; 10 execution failures. One of the failed runs also retained a matching candidate, so total candidate coverage is 67/134. Per-node logs recover matching boxes in two additional aborted runs.",
        ),
        (
            "Target definition must be fixed first",
            f"{summary['potential_anchor_attribute_alignment_flags']} questions are flagged for a likely anchor/attribute mismatch. They combine an upper_body annotation with a detector prompt for a smaller part or accessory. These are review flags, not a claim that every annotation is wrong. In 0162_q02 the question asks for shoes, but its GT rectangle covers the child's torso. In 0162_q01 the target rectangle covers the girl while the detector searches only for trousers. In 2547_q02 we ask SAM3 for 'center' although the annotation encloses the full pinwheel. Separate grounding_target, queried_attribute, reference_entities and relations. Align localization with the existing annotation ontology; do not pass GT categories or answers to inference, change labels silently, or call an adjusted metric the original benchmark.",
        ),
        (
            "Final box selection is structurally missing",
            f"Every box inherits its region's score, then SAM3 confidence breaks ties. Confidence says how well a box matches a category, not which person/object satisfies the relationships. Even oracle region selection followed by this confidence rule reaches only {summary['oracle_region_selection_with_confidence_box_ranking_hits']}/134 on the fixed pool, whereas 67 questions already have a matching box. 0411_q01 is decisive: Qwen correctly describes the camera operator's red shirt, yet the output selects someone else's shirt. Ask Qwen to select a numbered candidate in the full region, with explicit references and an abstention option. This preserves the requested whole-region input.",
        ),
        (
            "Numbered boxes help, but do not solve Qwen's limitations",
            f"On {probe_summary['original_winning_region_cases']} selected failures using their original winning regions, numbered-region selection recovered {probe_summary['numbered_region_hits']}; adding an overview recovered {probe_summary['overview_plus_region_hits']}. These are diagnostic cases, not a held-out accuracy estimate. 0411_q01, 0739_q01 and 0860_q02 were recovered with region-only numbered boxes. 0719_q02 still chose the wrong umbrella even though the correct umbrella was clearly present. On two separate oracle-correct-region probes, the clock could be selected but the boy's shirt in 0346_q02 remained wrong. A larger verifier may be worth testing after the task alignment is corrected; extra overview context was not uniformly beneficial to the 2B model.",
        ),
        (
            "Several SAM3 misses are prompt or threshold errors",
            "The exact same clock crop scored IoU 0.662 with prompt 'face' and 0.974 with 'clock'; the best clock proposal's confidence increased from about 0.09 to 0.89. For the plant case, 'pot' gave retained best IoU 0.301 while 'plant pot' gave 0.663. For the pinwheel, 'center' returned nothing and 'pinwheel' reached IoU 0.979. A rolled object already had a 0.934-IoU proposal at confidence 0.0185; the bicycle already had a 0.923-IoU proposal at confidence 0.022. The global 0.1 threshold discarded them. Preserve parent nouns and use a recall-oriented proposal stage followed by relational validation. Raising the cap alone recovered no additional IoU-0.5 hits at the existing threshold in these selected probes, despite many capped production regions. Lower thresholds also create false positives and must not replace grounding.",
        ),
        (
            "The score should separate semantic evidence from proposal health",
            f"{summary['high_score_no_target_nodes']} crops with Qwen score at least 0.8 exclude the annotated target, affecting {summary['questions_with_high_score_no_target']} questions. Conversely, target-containing crops sometimes lose the reference needed for the relation, so a low score is not automatically a VLM error. {summary['zero_qwen_but_reward_above_0_2_nodes']} regions with Qwen score exactly zero nevertheless received reward above 0.2 from detector terms. Box-count reward saturated in {summary['count_reward_saturated_nodes']} regions, and median diversity was {summary['median_diversity_score']:.3f}. Use count/diversity as proposal-health or exploration signals, not proof of query relevance. On the fixed pool, simple Qwen-times-confidence reranking gives 30 hits versus 27; this small post-hoc gain shows weight changes alone are insufficient.",
        ),
        (
            "Search and clustering need controlled ablations",
            f"{summary['fallback_splits']}/{summary['splits']} splits ({100 * summary['fallback_splits'] / summary['splits']:.1f}%) rejected the semantic partition and used spatial tiles. The PCA-color grouping therefore adds work without reliably producing semantic regions. Test overlapping spatial tiles and full DINO features with adjacency constraints under the same model-call budget. Median root exploration bonus was {summary['median_root_exploration_bonus']:.3f}, median backed-up quality {summary['median_selected_root_quality']:.3f}, and median time penalty {summary['median_root_cost_penalty']:.3f}; the cost term is not the dominant measured term. Forty-five misses had a finer target-containing crop left unvisited, but this is only a post-hoc opportunity measure, not proof that more budget would solve them. Compare UCT, lower exploration and uncertainty-driven zoom on a held-out subset before changing the full search.",
        ),
        (
            "Execution fixes applied and tested",
            "The three aspect failures came from spatial fallback tiles tightened around sparse masks: two runs produced a 620x3 crop, another a 461x2 crop. Fallback children now retain the full rectangular tile extent and preserve the active mask separately. Seven other runs stopped on invalid Qwen JSON. A dedicated recoverable response exception now saves raw failed replies, records a warning and continues other regions without backing up an invented zero score. Unexpected runtime/value errors return accumulated state so valid earlier evidence is retained. Regression tests cover pixel preservation, thin-tile geometry, malformed-response logging and continued search. The old 134-question run has not been rerun with these fixes, so its reported accuracy is unchanged.",
        ),
        (
            "Recommended order",
            "1. Confirm and implement the grounding-target ontology separately from the requested attribute.\n2. Ground an explicit candidate ID or point using the full region and relation/anchor checks.\n3. Repair concept extraction and retain low-confidence candidates for validation; save pre-filter box scores.\n4. Compare simpler overlapping tiles and query/uncertainty-driven zoom against the current clustering and UCT.\n5. Rerun the ten execution failures with the tested robustness fixes, then validate the architectural changes on a held-out question split before another full benchmark.",
        ),
    ]
    markdown = "# Failure audit and evidence-based improvements\n\n" + "\n\n".join(
        f"## {title}\n\n{text}" for title, text in sections
    )
    (output / "findings.md").write_text(markdown)
    body = "".join(
        f'<h2>{html.escape(title)}</h2><p style="white-space:pre-line">{html.escape(text)}</p>'
        for title, text in sections
    )
    examples = " ".join(
        f'<a href="visuals/{qid}.jpg">{qid}</a>'
        for qid in (
            "0162_q02",
            "0162_q01",
            "0411_q01",
            "0719_q02",
            "0860_q02",
            "1198_q02",
            "1537_q02",
            "2547_q02",
            "3194_q02",
            "1394_q02",
        )
    )
    (output / "findings.html").write_text(
        '<html><meta charset="utf-8"><style>body{font:18px system-ui;line-height:1.55;max-width:1200px;margin:35px auto;padding:15px}a{margin-right:12px}</style><h1>Failure audit and improvements</h1><nav><a href="index.html">All 134 traces</a><a href="visuals/index.html">All review cards</a><a href="audit.json">Audit data</a><a href="probes/grounding_probes.json">Grounding probes</a><a href="probes/detector_probes.json">SAM probes</a></nav>'
        + body
        + "<h2>Case evidence</h2>"
        + examples
        + "</html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("new_outputs/sam3_region_qwen_all")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/data_new"))
    args = parser.parse_args()
    rows, summary = audit(args.run_root, args.data_dir)
    output = write_report(args.run_root, rows, summary)
    print(json.dumps(summary, indent=2))
    print(f"Report: {output}/index.html")


if __name__ == "__main__":
    main()
