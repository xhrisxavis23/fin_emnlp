from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


@dataclass(eq=True, frozen=True)
class UndirectedNode:
    content: str
    label: str = ""
    id: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.id:
            return
        object.__setattr__(self, "id", f"{self.label}:{hash((self.label, self.content))}")


class UndirectedGraph:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._nodes: dict[str, UndirectedNode] = {}
        self._adj: dict[str, set[str]] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            with self.path.open("rb") as f:
                data = pickle.load(f)
            self._nodes = data.get("nodes", {})
            self._adj = data.get("adj", {})
        except Exception:
            self._nodes = {}
            self._adj = {}

    def _dump(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump({"nodes": self._nodes, "adj": self._adj}, f)

    def size(self) -> int:
        return len(self._nodes)

    def add_node(self, *, node: UndirectedNode) -> None:
        if node.id not in self._nodes:
            self._nodes[node.id] = node
            self._adj.setdefault(node.id, set())
            self._dump()

    def add_nodes(self, *, node: UndirectedNode, neighbors: Iterable[UndirectedNode]) -> None:
        self.add_node(node=node)
        for nb in neighbors or []:
            self.add_node(node=nb)
            self._adj[node.id].add(nb.id)
            self._adj[nb.id].add(node.id)
        self._dump()

    def get_node_by_content(self, *, content: str) -> UndirectedNode | None:
        for node in self._nodes.values():
            if node.content == content:
                return node
        return None

    def get_all_nodes_by_label(self, label: str) -> list[UndirectedNode]:
        return [n for n in self._nodes.values() if n.label == label]

    def get_all_nodes_by_label_list(self, labels: Iterable[str]) -> list[UndirectedNode]:
        """
        Backward-compatible helper for callers that pass multiple labels.

        Some CoSTEER knowledge-management utilities expect `get_all_nodes_by_label_list(["a", "b"])`.
        """
        label_set = set(labels or [])
        if not label_set:
            return []
        return [n for n in self._nodes.values() if n.label in label_set]

    def _bfs(self, start_id: str, step: int, *, block_labels: set[str] | None = None) -> dict[str, int]:
        dist: dict[str, int] = {start_id: 0}
        q: list[str] = [start_id]
        while q:
            cur = q.pop(0)
            if dist[cur] >= step:
                continue
            for nb in self._adj.get(cur, set()):
                if nb in dist:
                    continue
                if block_labels and self._nodes[nb].label not in block_labels and nb != start_id:
                    continue
                dist[nb] = dist[cur] + 1
                q.append(nb)
        return dist

    def query_by_node(
        self,
        *,
        node: UndirectedNode,
        step: int = 1,
        constraint_labels: list[str] | None = None,
        constraint_node: UndirectedNode | None = None,
        constraint_distance: float = 0,
        block: bool = False,
    ) -> list[UndirectedNode]:
        if node.id not in self._nodes:
            return []
        block_labels = set(constraint_labels or []) if block else None
        dist = self._bfs(node.id, step, block_labels=block_labels)
        out = [self._nodes[nid] for nid, d in dist.items() if d > 0]
        if constraint_labels:
            out = [n for n in out if n.label in set(constraint_labels)]
        if constraint_node and constraint_node.id in self._nodes and constraint_distance > 0:
            keep: list[UndirectedNode] = []
            cdist = self._bfs(constraint_node.id, int(constraint_distance))
            for n in out:
                if n.id in cdist:
                    keep.append(n)
            out = keep
        return out

    def query_by_content(
        self,
        *,
        content: str | list[str],
        topk_k: int = 5,
        step: int = 1,
        constraint_labels: list[str] | None = None,
        constraint_node: UndirectedNode | None = None,
        similarity_threshold: float = 0.0,
        constraint_distance: float = 0,
        block: bool = False,
    ) -> list[UndirectedNode]:
        queries = [content] if isinstance(content, str) else list(content or [])
        candidates = list(self._nodes.values())
        if constraint_labels and block:
            allowed = set(constraint_labels)
            candidates = [n for n in candidates if n.label in allowed]
        scored: list[tuple[float, UndirectedNode]] = []
        for q in queries:
            for n in candidates:
                score = _sim(q, n.content)
                if score >= similarity_threshold:
                    scored.append((score, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [n for _, n in scored[:topk_k]]
        if step > 0:
            expanded: dict[str, UndirectedNode] = {n.id: n for n in top}
            for n in top:
                for nb in self.query_by_node(
                    node=n,
                    step=step,
                    constraint_labels=constraint_labels,
                    constraint_node=constraint_node,
                    constraint_distance=constraint_distance,
                    block=block,
                ):
                    expanded.setdefault(nb.id, nb)
            top = list(expanded.values())
        return top

    def get_nodes_intersection(
        self,
        node_list: list[UndirectedNode],
        *,
        steps: int = 1,
        constraint_labels: list[str] | None = None,
    ) -> list[UndirectedNode]:
        if not node_list:
            return []
        neighborhoods: list[set[str]] = []
        for n in node_list:
            if n.id not in self._nodes:
                return []
            dist = self._bfs(n.id, steps)
            neighborhoods.append(set(dist.keys()))
        inter = set.intersection(*neighborhoods)
        inter.discard(node_list[0].id)
        out = [self._nodes[i] for i in inter if i in self._nodes]
        if constraint_labels:
            allowed = set(constraint_labels)
            out = [n for n in out if n.label in allowed]
        return out
