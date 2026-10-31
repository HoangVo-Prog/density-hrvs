"""Standalone HTML viewer for one manual run, including partial/live outputs."""

from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

from ..core.contracts import SearchConfig, SearchReport
from ..core.events import RunEvent


class RunRequest(TypedDict):
    """User-selected inference inputs; no ground-truth answer/box is included."""

    question: str
    query: str
    detector_prompt: str
    config: SearchConfig
    image_source: str  # Image path or dataset ID, for reproducibility.
    image_id: NotRequired[str]
    question_id: NotRequired[str]


class RunStatus(TypedDict):
    """Lifecycle status persisted even when weights fail to load or a run is cancelled."""

    state: Literal["running", "completed", "failed", "cancelled"]
    started_at: str
    updated_at: str
    error: NotRequired[str]
    result_path: NotRequired[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_events(output: Path) -> list[RunEvent]:
    """Tolerate a partially written final JSONL line during live inspection."""
    path = output / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(cast(RunEvent, json.loads(line)))
        except json.JSONDecodeError:
            continue
    return events


def render_run_page(output: Path, request: RunRequest, status: RunStatus) -> None:
    """Render available artifacts without requiring a successful final result."""
    status["updated_at"] = utc_now()
    (output / "status.json").write_text(json.dumps(status, indent=2))
    events = read_events(output)
    result_file = output / "results.json"
    result = (
        cast(SearchReport, json.loads(result_file.read_text()))
        if result_file.exists()
        else None
    )

    def link(path: str, label: str) -> str:
        return (
            f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>'
            if (output / path).exists()
            else ""
        )

    def figure(path: str, label: str) -> str:
        if not (output / path).exists():
            return ""
        url = html.escape(path, quote=True)
        return f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{url}"><img loading="lazy" src="{url}"></a></figure>'

    nav = " | ".join(
        filter(
            None,
            [
                link("run.log", "Readable log"),
                link("events.jsonl", "Structured events"),
                link("request.json", "Inputs/configuration"),
                link("results.json", "Results JSON"),
                link("stages/tree/index.html", "Tree and Q values"),
                link(
                    "inspection/index.html", "Ground-truth inspection (post-run only)"
                ),
            ],
        )
    )
    overview = figure(
        "source_preview.jpg", "Original image — no annotations were given to the models"
    )
    root = "stages/nodes/node_0000"
    initial = "".join(
        figure(f"{root}/{name}", label)
        for name, label in [
            ("dino_pca.png", "DINO PCA features used for clustering"),
            ("slic_selected_boundaries.jpg", "Selected SLIC boundaries"),
            (
                "kmeans_colors.png",
                "K-means groups — arbitrary colors, not relevance scores",
            ),
            (
                "selected_children.jpg",
                "Child-region rectangles in the root; labels here are zero-based child slots",
            ),
        ]
    )
    initial += link(
        f"{root}/index.html", "All initial DINO/SLIC/K-means stages and numeric arrays"
    )
    sections = []
    for event in events:
        if event["event"] != "iteration_started":
            continue
        data = event["data"]
        nid = data.get("node_id", -1)
        folder = f"stages/nodes/node_{nid:04d}"
        scores = output / folder / "region_score.json"
        components = output / folder / "region_score_components.json"
        score = json.loads(scores.read_text()) if scores.exists() else {}
        parts = json.loads(components.read_text()) if components.exists() else {}
        tables = ""
        if parts:
            tables = (
                "<table><tr><th>Best candidate Qwen</th><th>Mean verified SAM3 confidence</th><th>Count reward</th><th>Diversity</th><th>Combined region reward</th><th>Gate</th><th>Checked</th><th>Verified</th></tr><tr>"
                + "".join(
                    f"<td>{parts.get(k, 0):.4f}</td>"
                    for k in (
                        "qwen_score",
                        "mean_confidence",
                        "count_score",
                        "diversity_score",
                        "combined_score",
                    )
                )
                + f"<td>{html.escape(str(parts.get('gate', 'legacy')))}</td>"
                + f"<td>{parts.get('candidate_checks', 0)}</td>"
                + f"<td>{parts.get('verified_box_count', 0)}</td></tr></table>"
            )
        explain = html.escape(
            str(score.get("reason", "No valid Qwen score saved yet."))
        )
        pics = figure(
            f"{folder}/selected_region_scene.jpg",
            "CYAN rectangle = selected REGION in the full image; not an object prediction",
        )
        pics += figure(
            f"{folder}/sam3_distinct_boxes.jpg",
            "ORANGE boxes = SAM3 candidates after NMS; Qwen verifies the top candidates on separate evidence sheets",
        )
        exact = figure(
            f"{folder}/region_input.png",
            "Exact unmasked region supplied to SAM3 (display scaled); Qwen receives candidate context/detail sheets",
        )
        failures = link(
            f"{folder}/invalid_score_response.json",
            "Invalid raw model replies and recovery details",
        )
        stage_links = " | ".join(
            filter(
                None,
                [
                    link(f"{folder}/index.html", "All stages for this node"),
                    link(
                        f"{folder}/sam3_all_candidates.json",
                        "All native SAM3 boxes and filtering decisions",
                    ),
                    link(f"{folder}/detections.json", "Boxes retained before NMS"),
                    link(
                        f"{folder}/candidate_checks.json",
                        "Candidate verification results and coordinate mappings",
                    ),
                    failures,
                ],
            )
        )
        detection_state = (
            ""
            if (output / folder / "detections.json").exists()
            else '<p class="note">No SAM3 output saved for this node: it is still running or the region evaluation did not finish. See the event log.</p>'
        )
        sections.append(
            f'<section id="node-{nid}"><h2>Iteration {data.get("iteration")} — region N{nid}, depth {data.get("depth")}</h2><p>Tree path: {html.escape(str(data.get("path")))}. Global region bbox: {html.escape(str(data.get("bbox")))}.</p>{tables}<p>{explain}</p><div class="grid">{pics}</div>{detection_state}<details><summary>Exact whole-region input</summary>{exact}</details><p>{stage_links}</p></section>'
        )
    log = []
    for event in events:
        data = event["data"]
        branch = ""
        if data.get("candidates"):
            lines = "".join(
                f"<tr><td>N{c['node_id']}</td><td>{c['visits']}</td><td>{c['quality']:.4f}</td><td>{c['exploration_bonus']:.4f}</td><td>{c['cost_penalty']:.4f}</td><td>{c['selection_index']:.4f}</td></tr>"
                for c in data.get("candidates", [])
            )
            branch = f"<table><tr><th>Eligible child</th><th>Visits</th><th>Q</th><th>Exploration</th><th>Time penalty</th><th>Selection index</th></tr>{lines}</table>"
        log.append(
            f'<details class="event"><summary>#{event["sequence"]} +{event["elapsed_seconds"]:.2f}s — {html.escape(event["message"])}</summary>{branch}<pre>{html.escape(json.dumps(data, indent=2))}</pre></details>'
        )
    final = ""
    if result is not None:
        final = f"<p>Stop: {html.escape(result['stop_reason'])}. Local Qwen generations: {result['usage'].get('local_score_calls', 0)}. Retained object boxes: {len(result['proposals'])}.</p>"
        final += figure(
            "preview.jpg",
            "Final object proposals — rank uses per-candidate Qwen verification fused with SAM3 confidence",
        )
        final += "<p>Box confidence is category confidence; Qwen scores are ordinal match ratings, not calibrated probabilities.</p>"
    error = html.escape(status.get("error", ""))
    page = f"""<!doctype html><html><meta charset="utf-8"><title>Visual search run</title>
<style>body{{font:17px system-ui;line-height:1.5;max-width:1600px;margin:25px auto;padding:18px;background:#101827;color:#eee}}a{{color:#7dd3fc}}img{{max-width:100%;max-height:650px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:12px}}figure{{margin:12px 0;padding:10px;border:1px solid #64748b}}figcaption{{margin-bottom:8px}}section{{border-top:2px solid #64748b;margin-top:30px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #64748b}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#1f2937;padding:12px}}details{{padding:8px;margin:8px 0;border:1px solid #475569}}.note{{color:#fbbf24}}input{{padding:10px;width:90%;font:inherit}}</style>
<h1>Single-image visual search — {status["state"]}</h1><p><b>Question:</b> {html.escape(request["question"])}</p><p><b>Search description:</b> {html.escape(request["query"])}</p><p><b>SAM3 concept:</b> {html.escape(request["detector_prompt"])}. <b>Input:</b> {html.escape(request["image_source"])}</p><p>{nav}</p><p class="note">Region selection and object-box selection are separate. CYAN = selected region. ORANGE = SAM3 object boxes. Ground truth appears only in the separate post-run inspection when a dataset question is selected.</p><p>Refresh this page to see new steps while the run is active. Last updated: {status["updated_at"]}.</p><p>{error}</p>{overview}<h2>Initial partition</h2><div class="grid">{initial}</div>{"".join(sections)}<section><h2>Final output</h2>{final}</section><section><h2>Step-by-step decisions</h2><input id="filter" placeholder="Filter events, e.g. branch, N2, error, reward"><div id="events">{"".join(log)}</div></section>
<script>document.getElementById('filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('.event').forEach(x=>x.style.display=x.textContent.toLowerCase().includes(q)?'':'none')}});</script></html>"""
    temporary = output / "index.tmp"
    temporary.write_text(page)
    temporary.replace(output / "index.html")
