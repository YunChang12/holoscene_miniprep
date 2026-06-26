"""Simple scene graph inference for HoloScene."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .writer import write_json


def _xy_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0, ay0, _ = a["bbox_min"]
    ax1, ay1, _ = a["bbox_max"]
    bx0, by0, _ = b["bbox_min"]
    bx1, by1, _ = b["bbox_max"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1e-8, (ax1 - ax0) * (ay1 - ay0))
    return float(inter / area_a)


def infer_graph_simple(
    bbox_report: dict[str, Any],
    vertical_gap_threshold: float = 0.08,
    xy_overlap_threshold: float = 0.15,
    root_id: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Infer a small support graph from instance bboxes."""

    objects = bbox_report.get("objects", [])
    node_ids = [int(obj["holoscene_node_id"]) for obj in objects]
    parent: dict[int, int] = {node: int(root_id) for node in node_ids}
    debug = {"decisions": []}
    for child in objects:
        child_id = int(child["holoscene_node_id"])
        best = None
        best_score = -math.inf
        candidates = []
        for cand in objects:
            cand_id = int(cand["holoscene_node_id"])
            if cand_id == child_id:
                continue
            gap = float(child["z_min"]) - float(cand["z_max"])
            overlap = _xy_overlap(child, cand)
            score = overlap - max(gap, 0.0)
            candidate = {
                "parent": cand_id,
                "vertical_gap": float(gap),
                "xy_overlap": float(overlap),
                "score": float(score),
                "accepted": bool(0.0 <= gap <= float(vertical_gap_threshold) and overlap >= float(xy_overlap_threshold)),
            }
            candidates.append(candidate)
            if 0.0 <= gap <= float(vertical_gap_threshold) and overlap >= float(xy_overlap_threshold):
                if score > best_score:
                    best = cand_id
                    best_score = score
        if best is not None:
            parent[child_id] = int(best)
            reason = "parent below child and xy overlap is high"
        else:
            reason = "default_root: no confident support candidate"
        debug["decisions"].append(
            {
                "child": child_id,
                "node_id": child_id,
                "chosen_parent": parent[child_id],
                "parent": parent[child_id],
                "candidates": sorted(candidates, key=lambda x: x["score"], reverse=True),
                "reason": reason,
            }
        )
    adjacency: dict[int, set[int]] = {int(root_id): set()}
    for node in node_ids:
        adjacency.setdefault(node, set())
    for child, par in parent.items():
        adjacency.setdefault(par, set()).add(child)
        adjacency.setdefault(child, set()).add(par)
    graph = [{"node_id": int(node), "adj_nodes": sorted(int(v) for v in neighbors)} for node, neighbors in sorted(adjacency.items())]
    debug["objects"] = objects
    debug["parents"] = {str(k): int(v) for k, v in parent.items()}
    debug["root_id"] = int(root_id)
    return graph, debug


def write_graph_json(scene_dir: str | Path, graph: list[dict[str, Any]]) -> Path:
    """Write graph.json."""

    return write_json(Path(scene_dir) / "graph.json", graph)


def write_graph_debug(scene_dir: str | Path, debug: dict[str, Any]) -> Path:
    """Write meta/graph_debug.json."""

    return write_json(Path(scene_dir) / "meta" / "graph_debug.json", debug)


def visualize_graph(scene_dir: str | Path, graph: list[dict[str, Any]]) -> Path:
    """Create a simple graph_vis.png without graphviz."""

    out = Path(scene_dir) / "review" / "graph_vis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    nodes = [int(item["node_id"]) for item in graph]
    w, h = 900, max(260, 120 + 70 * len(nodes))
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    positions = {}
    for idx, node in enumerate(nodes):
        x = 120 + (idx % 5) * 170
        y = 80 + (idx // 5) * 120
        positions[node] = (x, y)
    drawn_edges = set()
    for item in graph:
        a = int(item["node_id"])
        for b in item.get("adj_nodes", []):
            b = int(b)
            key = tuple(sorted((a, b)))
            if key in drawn_edges or b not in positions:
                continue
            drawn_edges.add(key)
            draw.line([positions[a], positions[b]], fill=(80, 80, 80), width=2)
    for node, (x, y) in positions.items():
        fill = (220, 235, 255) if node == 0 else (235, 255, 220)
        draw.ellipse([x - 34, y - 34, x + 34, y + 34], fill=fill, outline=(20, 20, 20), width=2)
        draw.text((x - 18, y - 8), str(node), fill=(0, 0, 0))
    img.save(out)
    return out
