"""Cost-aware UCT over a finite, lazily generated image-region tree.

This is a visual-search adaptation of Agent-UCT: complete crop evaluations back
up raw quality, while expected remaining execution time only affects selection.
It does not inherit the paper's fixed-workflow regret guarantee.
"""

from collections.abc import Callable
from dataclasses import dataclass
import math
import time
from typing import Literal, TypedDict

from ..core.contracts import (
    ChildRegion,
    NodeEvidence,
    SearchConfig,
    SearchNode,
    SearchResult,
    SearchStep,
    SearchUsage,
)
from ..core.errors import (
    APIBudgetExceeded,
    InvalidScoreResponse,
    LocalBudgetExceeded,
)
from ..core.events import BackupUpdate, BranchCandidate, EventSink, ignore_event
from .config import default_search_config, validate_search_config

__all__ = [
    "APIBudgetExceeded",
    "LocalBudgetExceeded",
    "Selection",
    "default_search_config",
    "explore_tree",
    "new_node",
    "remaining_cost",
    "select_child",
    "validate_search_config",
]


class Selection(TypedDict):
    """Pre-decision terms for selecting one child subtree."""

    quality: float
    exploration_bonus: float
    estimated_remaining_seconds: float
    selection_index: float


NodeEvaluator = Callable[[SearchNode], NodeEvidence]
RegionSplitter = Callable[[SearchNode], list[ChildRegion]]
CostEstimator = Callable[[SearchNode], float]


def new_node(
    region: ChildRegion, node_id: int, parent_id: int, depth: int
) -> SearchNode:
    """Create an unvisited allocation node without inventing a quality estimate."""
    return {
        **region,
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "children": [],
        "visits": 0,
        "reward_sum": 0.0,
        "evaluated": False,
        "exhausted": False,
    }


def remaining_cost(
    node_id: int, nodes: list[SearchNode], estimate: CostEstimator
) -> float:
    """Forecast the next complete evaluation under historical continuations.

    Expanded nodes average their unfinished children using smoothed visit counts;
    frontiers price all missing scoring, detection, and subdivision work.
    """
    node = nodes[node_id]
    active = [nodes[i] for i in node["children"] if not nodes[i]["exhausted"]]
    if not active:
        return max(0.0, estimate(node))
    denominator = sum(child["visits"] for child in active) + 1
    return sum(
        (child["visits"] + 1 / len(active))
        / denominator
        * remaining_cost(child["node_id"], nodes, estimate)
        for child in active
    )


def _active_children(parent: SearchNode, nodes: list[SearchNode]) -> list[SearchNode]:
    return [nodes[i] for i in parent["children"] if not nodes[i]["exhausted"]]


def _branch_terms(
    child: SearchNode,
    parent: SearchNode,
    nodes: list[SearchNode],
    config: SearchConfig,
    estimate: CostEstimator,
) -> tuple[Selection, BranchCandidate]:
    visits = child["visits"]
    quality = child["reward_sum"] / visits if visits else 0.0
    bonus = (
        0.0
        if not visits or config["strategy"] == "greedy"
        else (
            config["exploration_weight"]
            * math.sqrt(math.log(parent["visits"] + 1) / visits)
        )
    )
    cost = remaining_cost(child["node_id"], nodes, estimate)
    penalty = (
        config["cost_weight"] * cost / config["reference_cost_seconds"]
        if config["strategy"] == "agent-uct"
        else 0.0
    )
    selection: Selection = {
        "quality": quality,
        "exploration_bonus": bonus,
        "estimated_remaining_seconds": cost,
        "selection_index": quality + bonus - penalty,
    }
    candidate: BranchCandidate = {
        "node_id": child["node_id"],
        "visits": visits,
        "quality": quality,
        "exploration_bonus": bonus,
        "estimated_remaining_seconds": cost,
        "cost_penalty": penalty,
        "selection_index": selection["selection_index"],
    }
    return selection, candidate


def _selection_reason(
    unvisited: list[SearchNode],
    candidates: list[BranchCandidate],
    chosen: Selection,
) -> str:
    reason = (
        "Visit an unvisited child first"
        if unvisited
        else "Maximize Q + exploration bonus - time penalty"
    )
    ties = sum(
        abs(candidate["selection_index"] - chosen["selection_index"]) < 1e-12
        for candidate in candidates
    )
    if ties > 1:
        reason += "; equal priorities are broken by lowest node ID"
    return reason


def select_child(
    parent: SearchNode,
    nodes: list[SearchNode],
    config: SearchConfig,
    estimate: CostEstimator,
    on_event: EventSink = ignore_event,
) -> tuple[int, Selection]:
    """Choose an unfinished child, prioritizing unvisited children before UCT.

    Args:
        parent: Current allocation node and its completed-evaluation count.
        nodes: Node inventory indexed by node_id.
        config: Strategy, exploration coefficient, and normalized cost penalty.
        estimate: Read-only forecast of the unpaid work at a frontier.
        on_event: Optional decision logger.
    """
    active = _active_children(parent, nodes)
    if not active:
        raise ValueError("No unfinished child is available.")
    unvisited = [child for child in active if child["visits"] == 0]
    pool = unvisited or active
    terms: dict[int, Selection] = {}
    candidates: list[BranchCandidate] = []
    for child in pool:
        selection, candidate = _branch_terms(child, parent, nodes, config, estimate)
        terms[child["node_id"]] = selection
        candidates.append(candidate)
    # Unvisited quality is unknown: cost may order initial visits, never exclude one.
    chosen = max(
        pool,
        key=lambda child: (
            terms[child["node_id"]]["selection_index"],
            -child["node_id"],
        ),
    )
    reason = _selection_reason(unvisited, candidates, terms[chosen["node_id"]])
    eligible_ids = {node["node_id"] for node in pool}
    on_event(
        "branch_selected",
        f"Parent N{parent['node_id']} chose N{chosen['node_id']}: {reason}.",
        {
            "parent_id": parent["node_id"],
            "selected_node": chosen["node_id"],
            "candidates": candidates,
            "selection_reason": reason,
            "excluded_node_ids": [
                node["node_id"]
                for node in active
                if node["node_id"] not in eligible_ids
            ],
        },
    )
    return chosen["node_id"], terms[chosen["node_id"]]


def _add_children(
    parent: SearchNode,
    nodes: list[SearchNode],
    config: SearchConfig,
    split: RegionSplitter,
    on_event: EventSink,
) -> None:
    if parent["depth"] >= config["max_depth"]:
        parent["exhausted"] = True
        on_event(
            "split_skipped",
            f"N{parent['node_id']} reached maximum depth {config['max_depth']}.",
            {"node_id": parent["node_id"], "depth": parent["depth"]},
        )
        return
    for region in split(parent):
        child_id = len(nodes)
        nodes.append(new_node(region, child_id, parent["node_id"], parent["depth"] + 1))
        parent["children"].append(child_id)
    parent["exhausted"] = not parent["children"]
    on_event(
        "children_created",
        f"N{parent['node_id']} has {len(parent['children'])} child regions.",
        {
            "node_id": parent["node_id"],
            "child_ids": parent["children"].copy(),
            "child_bboxes": [nodes[i]["bbox"] for i in parent["children"]],
        },
    )


def _select_path(
    nodes: list[SearchNode],
    config: SearchConfig,
    estimate: CostEstimator,
    on_event: EventSink,
) -> tuple[list[int], Selection]:
    path = [0]
    selection: Selection = {
        "quality": 0.0,
        "exploration_bonus": 0.0,
        "estimated_remaining_seconds": estimate(nodes[0]),
        "selection_index": 0.0,
    }
    while nodes[path[-1]]["children"]:
        child_id, terms = select_child(
            nodes[path[-1]], nodes, config, estimate, on_event
        )
        if len(path) == 1:
            selection = terms
        path.append(child_id)
    return path, selection


def _backup_reward(
    nodes: list[SearchNode], path: list[int], reward: float
) -> list[BackupUpdate]:
    backups: list[BackupUpdate] = []
    for node_id in path:
        before_visits = nodes[node_id]["visits"]
        before_quality = (
            nodes[node_id]["reward_sum"] / before_visits if before_visits else 0.0
        )
        nodes[node_id]["visits"] += 1
        nodes[node_id]["reward_sum"] += reward
        backups.append(
            {
                "node_id": node_id,
                "visits_before": before_visits,
                "visits_after": nodes[node_id]["visits"],
                "quality_before": before_quality,
                "quality_after": nodes[node_id]["reward_sum"]
                / nodes[node_id]["visits"],
            }
        )
    return backups


def _refresh_exhaustion(nodes: list[SearchNode], path: list[int]) -> None:
    for node_id in reversed(path[:-1]):
        parent = nodes[node_id]
        parent["exhausted"] = bool(parent["children"]) and all(
            nodes[i]["exhausted"] for i in parent["children"]
        )


def _match_found(evidence: NodeEvidence, config: SearchConfig) -> bool:
    return any(
        proposal.get("candidate_verified", proposal["verifier"]["answer"] == "Yes")
        and proposal["score"] >= config["match_threshold"]
        for proposal in evidence["proposals"]
    )


StopReason = Literal[
    "exhausted", "api_budget", "local_budget", "iterations", "match", "error"
]


@dataclass
class _TreeRun:
    """Mutable state shared by one explore_tree call and its outcome handlers."""

    nodes: list[SearchNode]
    trace: list[SearchStep]
    config: SearchConfig
    split: RegionSplitter
    usage: SearchUsage
    on_step: Callable[[list[SearchNode], list[SearchStep]], None] | None
    on_event: EventSink

    def snapshot(self) -> None:
        """Publish the current tree and trace to the optional step observer."""
        if self.on_step is not None:
            self.on_step(self.nodes, self.trace)

    def expand(self, node: SearchNode) -> None:
        """Lazily create the children of ``node`` (or mark it exhausted)."""
        _add_children(node, self.nodes, self.config, self.split, self.on_event)

    def finish_step(self, step: SearchStep, started: float) -> None:
        """Stamp the final call count and duration onto ``step``."""
        step["calls_after"] = self.usage["api_calls"]
        step["elapsed_seconds"] = time.perf_counter() - started


def _api_budget_spent(config: SearchConfig, usage: SearchUsage) -> bool:
    """Return whether a configured HTTP call cap has been reached."""
    return bool(config["max_api_calls"]) and usage["api_calls"] >= config["max_api_calls"]


def _start_step(
    selection: Selection, iteration: int, node: SearchNode, path: list[int], calls: int
) -> SearchStep:
    """Create the trace record for one iteration before evaluation runs."""
    return {
        **selection,
        "iteration": iteration,
        "node_id": node["node_id"],
        "path": path,
        "calls_before": calls,
        "calls_after": calls,
        "completed": False,
        "raw_reward": None,
        "elapsed_seconds": 0.0,
    }


def _announce_iteration(
    run: _TreeRun,
    iteration: int,
    node: SearchNode,
    path: list[int],
    progress: Callable[[str], None],
) -> None:
    """Emit the structured and human-readable start-of-iteration messages."""
    run.on_event(
        "iteration_started",
        f"Iteration {iteration}: explore N{node['node_id']}, depth {node['depth']}, path {path}.",
        {
            "iteration": iteration,
            "node_id": node["node_id"],
            "path": path.copy(),
            "bbox": node["bbox"],
            "depth": node["depth"],
            "usage": run.usage.copy(),
        },
    )
    progress(
        f"Iteration {iteration}: region {node['node_id']}, depth {node['depth']}, HTTP calls {run.usage['api_calls']}"
    )


def _handle_budget_exhausted(
    run: _TreeRun,
    node: SearchNode,
    step: SearchStep,
    started: float,
    exc: APIBudgetExceeded,
) -> StopReason:
    """Record a budget-interrupted step; partial work never backs up a reward."""
    reason: StopReason = (
        "local_budget" if isinstance(exc, LocalBudgetExceeded) else "api_budget"
    )
    run.finish_step(step, started)
    run.trace.append(step)
    run.on_event(
        "budget_exhausted",
        str(exc),
        {
            "node_id": node["node_id"],
            "iteration": step["iteration"],
            "stop_reason": reason,
            "usage": run.usage.copy(),
        },
    )
    run.snapshot()
    return reason


def _handle_invalid_score(
    run: _TreeRun,
    node: SearchNode,
    path: list[int],
    step: SearchStep,
    started: float,
    exc: InvalidScoreResponse,
) -> None:
    """Treat an unparseable verifier reply as uncertainty, not a zero reward.

    The node is still expanded so its children can yield valid evidence;
    terminal failed leaves become exhausted.
    """
    node["evaluation_error"] = str(exc)
    step["evaluation_error"] = str(exc)
    run.finish_step(step, started)
    run.on_event(
        "evaluation_failed",
        f"N{node['node_id']}: invalid verifier reply; continue without backing up a zero.",
        {
            "node_id": node["node_id"],
            "iteration": step["iteration"],
            "recoverable": True,
            "error": str(exc),
            "raw_responses": exc.responses,
        },
    )
    run.expand(node)
    _refresh_exhaustion(run.nodes, path)
    run.trace.append(step)
    run.snapshot()


def _handle_execution_error(
    run: _TreeRun,
    node: SearchNode,
    step: SearchStep,
    started: float,
    exc: Exception,
) -> None:
    """Record a fatal evaluation failure; earlier evidence is retained."""
    run.finish_step(step, started)
    run.trace.append(step)
    run.on_event(
        "evaluation_failed",
        f"N{node['node_id']}: execution failed; retain earlier evidence.",
        {
            "node_id": node["node_id"],
            "iteration": step["iteration"],
            "recoverable": False,
            "error": str(exc),
        },
    )


def _handle_completed(
    run: _TreeRun,
    node: SearchNode,
    path: list[int],
    step: SearchStep,
    started: float,
    evidence: NodeEvidence,
) -> StopReason | None:
    """Back up a complete evaluation, expand the node, and decide whether to stop.

    Returns:
        ``"match"`` or ``"api_budget"`` when the search must stop, else None.
    """
    node["assessment"] = evidence["assessment"]
    node["evaluated"] = True
    reward = evidence["raw_reward"]
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError("A complete evaluation must return raw quality in [0, 1].")
    backups = _backup_reward(run.nodes, path, reward)
    run.on_event(
        "reward_backed_up",
        f"Backed up region reward {reward:.4f} along path {path}.",
        {
            "node_id": node["node_id"],
            "iteration": step["iteration"],
            "reward": reward,
            "path": path.copy(),
            "backups": backups,
        },
    )
    step.update(calls_after=run.usage["api_calls"], completed=True, raw_reward=reward)

    stop: StopReason | None = None
    if run.config["stop_on_match"] and _match_found(evidence, run.config):
        stop = "match"
    elif _api_budget_spent(run.config, run.usage):
        stop = "api_budget"
    else:
        run.expand(node)
    _refresh_exhaustion(run.nodes, path)

    step["elapsed_seconds"] = time.perf_counter() - started
    run.trace.append(step)
    run.on_event(
        "iteration_completed",
        f"Iteration {step['iteration']} completed with reward {reward:.4f}.",
        {
            "node_id": node["node_id"],
            "iteration": step["iteration"],
            "reward": reward,
            "duration_seconds": step["elapsed_seconds"],
            "usage": run.usage.copy(),
        },
    )
    run.snapshot()
    return stop


def _run_iterations(
    run: _TreeRun, estimate: CostEstimator, evaluate: NodeEvaluator, progress: Callable[[str], None]
) -> tuple[StopReason, str]:
    """Select, evaluate, and back up frontiers until a stopping rule fires.

    Returns:
        The stopping reason and, for fatal failures only, the error message.
    """
    root = run.nodes[0]
    for index in range(run.config["max_iterations"]):
        if root["exhausted"]:
            return "exhausted", ""
        if _api_budget_spent(run.config, run.usage):
            return "api_budget", ""
        iteration = index + 1
        path, selection = _select_path(run.nodes, run.config, estimate, run.on_event)
        node = run.nodes[path[-1]]
        _announce_iteration(run, iteration, node, path, progress)
        started = time.perf_counter()
        step = _start_step(selection, iteration, node, path, run.usage["api_calls"])
        try:
            evidence = evaluate(node)
        except APIBudgetExceeded as exc:
            return _handle_budget_exhausted(run, node, step, started, exc), ""
        except InvalidScoreResponse as exc:
            _handle_invalid_score(run, node, path, step, started, exc)
            continue
        except (RuntimeError, ValueError) as exc:
            _handle_execution_error(run, node, step, started, exc)
            return "error", str(exc)
        stop = _handle_completed(run, node, path, step, started, evidence)
        if stop is not None:
            return stop, ""
    return ("exhausted" if root["exhausted"] else "iterations"), ""


def explore_tree(
    root: ChildRegion,
    query: str,
    config: SearchConfig,
    evaluate: NodeEvaluator,
    split: RegionSplitter,
    estimate: CostEstimator,
    usage: SearchUsage,
    progress: Callable[[str], None] = print,
    on_step: Callable[[list[SearchNode], list[SearchStep]], None] | None = None,
    on_event: EventSink = ignore_event,
) -> SearchResult:
    """Expand unique frontiers and back up only complete, raw quality returns.

    Args:
        root: Full-image bounds and active mask.
        query: Target description recorded with all observations.
        config: Validated finite search controls.
        evaluate: Assess a node and its proposal crops; may raise APIBudgetExceeded.
        split: Produce strictly smaller semantic/spatial children in global coordinates.
        estimate: Forecast remaining seconds from observed operation costs and caches.
        usage: Shared execution ledger updated by the evaluator's request observer.
        progress: Short status callback; receives no credentials or image payloads.
        on_step: Optional observer for each tree snapshot.
        on_event: Optional structured decision logger.

    Returns:
        Tree, trace, and stopping reason. The runner attaches cached observations.
    """
    validate_search_config(config)
    started = time.perf_counter()
    run = _TreeRun(
        nodes=[new_node(root, 0, -1, 0)],
        trace=[],
        config=config,
        split=split,
        usage=usage,
        on_step=on_step,
        on_event=on_event,
    )
    run.expand(run.nodes[0])
    run.snapshot()
    # If the image cannot be partitioned, evaluate its root once rather than fail.
    if not run.nodes[0]["children"]:
        run.nodes[0]["exhausted"] = False

    stop_reason, error = _run_iterations(run, estimate, evaluate, progress)
    return {
        "query": query,
        "config": config,
        "nodes": run.nodes,
        "trace": run.trace,
        "ranked_regions": [],
        "ranked_clusters": [],
        "usage": usage,
        "stop_reason": stop_reason,
        "error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "warnings": [
            f"Node {node['node_id']}: {node['evaluation_error']}"
            for node in run.nodes
            if "evaluation_error" in node
        ],
    }
