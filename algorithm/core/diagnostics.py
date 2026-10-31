"""Bounded visual artifacts plus lossless numeric outputs for every search stage."""

import html
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import label
from skimage.segmentation import find_boundaries

from .contracts import ChildRegion, SearchNode, SearchStep


class ClusterQuality(TypedDict):
    """Geometric checks for a search partition, independent of target annotations."""

    bbox: tuple[int, int, int, int]  # Global bounds of this child.
    occupancy: float  # Active pixels divided by bbox area.
    aspect: float  # Larger bbox dimension divided by smaller one.
    components: int  # Number of 4-connected components on the diagnostic grid.
    largest_component_fraction: (
        float  # Pixels in the largest component / active pixels.
    )


def cluster_quality(child: ChildRegion) -> ClusterQuality:
    mask = child["mask"]
    x1, y1, x2, y2 = child["bbox"]
    # Geometry is measured exactly; fragmentation is bounded for very large images.
    scaled = Image.fromarray(mask).copy()
    scaled.thumbnail((512, 512), Image.Resampling.NEAREST)
    components = np.zeros(np.asarray(scaled).shape, dtype=np.int32)
    label(np.asarray(scaled), output=components)
    count = int(components.max())
    sizes = np.bincount(components.ravel())[1:]
    return {
        "bbox": child["bbox"],
        "occupancy": float(mask.mean()),
        "aspect": max(x2 - x1, y2 - y1) / max(1, min(x2 - x1, y2 - y1)),
        "components": int(count),
        "largest_component_fraction": float(sizes.max() / sizes.sum())
        if sizes.size
        else 0.0,
    }


def save_image(path: Path, pixels: NDArray, max_side: int = 1536) -> None:
    """Save a display derivative; numeric arrays are saved separately when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels)
    image.thumbnail((max_side, max_side))
    image.save(path)


def save_labels(folder: Path, name: str, labels: NDArray, rgb: NDArray) -> None:
    """Save lossless IDs, colored labels and RGB boundary overlays."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}.npz", labels=labels.astype(np.int32))
    codes = labels.astype(np.uint32)
    colors = np.stack(
        ((codes * 73 + 41) % 256, (codes * 151 + 83) % 256, (codes * 199 + 127) % 256),
        axis=-1,
    ).astype(np.uint8)
    save_image(folder / f"{name}_colors.png", colors)
    image = np.asarray(
        Image.fromarray(rgb).resize((labels.shape[1], labels.shape[0]))
    ).copy()
    image[find_boundaries(labels, mode="outer")] = (255, 255, 0)
    save_image(folder / f"{name}_boundaries.jpg", image)


def save_tree(folder: Path, nodes: list[SearchNode], trace: list[SearchStep]) -> None:
    """Write each allocation snapshot and an expandable tree with selection terms."""
    folder.mkdir(parents=True, exist_ok=True)
    state = [
        {key: value for key, value in node.items() if key != "mask"} for node in nodes
    ]
    snapshot = {"nodes": state, "trace": trace}
    (folder / f"iteration_{len(trace):03d}.json").write_text(
        json.dumps(snapshot, indent=2)
    )
    (folder / "latest.json").write_text(json.dumps(snapshot, indent=2))

    def branch(node: SearchNode) -> str:
        quality = node["reward_sum"] / node["visits"] if node["visits"] else 0
        link = f"../nodes/node_{node['node_id']:04d}/index.html"
        available = (folder / link).exists()
        stages = f'<a href="{link}">stages</a>' if available else "(not explored)"
        children = "".join(branch(nodes[i]) for i in node["children"])
        return f"<li><details open><summary>N{node['node_id']} depth={node['depth']} visits={node['visits']} Q={quality:.3f} bbox={node['bbox']} exhausted={node['exhausted']} {stages}</summary><ul>{children}</ul></details></li>"

    rows = "".join(
        f"<tr><td>{s['iteration']}</td><td>{s['path']}</td><td>{s['quality']:.3f}</td><td>{s['exploration_bonus']:.3f}</td><td>{s['estimated_remaining_seconds']:.2f}</td><td>{s['selection_index']:.3f}</td><td>{s['raw_reward']}</td><td>{s['completed']}</td></tr>"
        for s in trace
    )
    (folder / "index.html").write_text(
        f'<html><meta charset="utf-8"><style>body{{font:16px system-ui}}td,th{{padding:7px;border:1px solid #ccc}}</style><h1>Recursive tree and allocation trace</h1><ul>{branch(nodes[0])}</ul><table><tr><th>Step</th><th>Path</th><th>Q</th><th>Exploration</th><th>Predicted seconds</th><th>Selection index</th><th>Reward</th><th>Completed</th></tr>{rows}</table></html>'
    )


def stage_index(folder: Path) -> None:
    """Build a gallery exposing all saved images and numeric stage records."""
    images = sorted(p for p in folder.rglob("*") if p.suffix in {".png", ".jpg"})
    figures = "".join(
        f'<figure><figcaption>{html.escape(str(p.relative_to(folder)))}</figcaption><a href="{p.relative_to(folder)}"><img loading="lazy" src="{p.relative_to(folder)}" style="max-width:100%"></a></figure>'
        for p in images
    )
    files = "".join(
        f'<li><a href="{p.relative_to(folder)}">{html.escape(str(p.relative_to(folder)))}</a></li>'
        for p in sorted(folder.rglob("*"))
        if p.suffix in {".json", ".npz"}
    )
    (folder / "index.html").write_text(
        f'<html><meta charset="utf-8"><style>body{{max-width:1500px;margin:auto;font:17px system-ui}}figure{{border:1px solid #aaa;padding:10px}}</style><h1>{html.escape(folder.name)} stages</h1><ul>{files}</ul>{figures}</html>'
    )
