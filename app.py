from __future__ import annotations

import heapq
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse


DIMS = 16
HOST = "0.0.0.0"
PORT = 8080


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: list[float]


DistFn = Callable[[list[float], list[float]], float]


def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


class BruteForce:
    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def knn(self, q: list[float], k: int, dist_fn: DistFn) -> list[tuple[float, int]]:
        results = [(dist_fn(q, item.emb), item.id) for item in self.items]
        results.sort(key=lambda pair: pair[0])
        return results[:k]

    def remove(self, item_id: int) -> None:
        self.items = [item for item in self.items if item.id != item_id]


@dataclass
class KDNode:
    item: VectorItem
    left: KDNode | None = None
    right: KDNode | None = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: KDNode | None = None

    def _insert(self, node: KDNode | None, item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item=item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(self.root, item, 0)

    def _knn(
        self,
        node: KDNode | None,
        q: list[float],
        k: int,
        depth: int,
        dist_fn: DistFn,
        heap: list[tuple[float, int]],
    ) -> None:
        if node is None:
            return

        distance = dist_fn(q, node.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-distance, node.item.id))
        elif distance < -heap[0][0]:
            heapq.heapreplace(heap, (-distance, node.item.id))

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def knn(self, q: list[float], k: int, dist_fn: DistFn) -> list[tuple[float, int]]:
        heap: list[tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        results = [(-distance, item_id) for distance, item_id in heap]
        results.sort(key=lambda pair: pair[0])
        return results

    def rebuild(self, items: list[VectorItem]) -> None:
        self.root = None
        for item in items:
            self.insert(item)


@dataclass
class HNSWNode:
    item: VectorItem
    max_layer: int
    nbrs: list[list[int]] = field(default_factory=list)


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.graph: dict[int, HNSWNode] = {}
        self.m = m
        self.m0 = 2 * m
        self.ef_build = ef_build
        self.m_l = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_point = -1
        self.rng = random.Random(42)

    def _rand_level(self) -> int:
        sample = max(self.rng.random(), 1e-12)
        return int(math.floor(-math.log(sample) * self.m_l))

    def _search_layer(
        self, q: list[float], ep: int, ef: int, layer: int, dist_fn: DistFn
    ) -> list[tuple[float, int]]:
        visited = {ep}
        candidates: list[tuple[float, int]] = []
        found: list[tuple[float, int]] = []

        start = self.graph[ep]
        start_distance = dist_fn(q, start.item.emb)
        heapq.heappush(candidates, (start_distance, ep))
        heapq.heappush(found, (-start_distance, ep))

        while candidates:
            current_distance, current_id = heapq.heappop(candidates)
            if len(found) >= ef and current_distance > -found[0][0]:
                break

            current = self.graph.get(current_id)
            if current is None or layer >= len(current.nbrs):
                continue

            for neighbor_id in current.nbrs[layer]:
                if neighbor_id in visited or neighbor_id not in self.graph:
                    continue
                visited.add(neighbor_id)
                neighbor_distance = dist_fn(q, self.graph[neighbor_id].item.emb)
                if len(found) < ef or neighbor_distance < -found[0][0]:
                    heapq.heappush(candidates, (neighbor_distance, neighbor_id))
                    heapq.heappush(found, (-neighbor_distance, neighbor_id))
                    if len(found) > ef:
                        heapq.heappop(found)

        results = [(-distance, item_id) for distance, item_id in found]
        results.sort(key=lambda pair: pair[0])
        return results

    @staticmethod
    def _select_neighbors(candidates: list[tuple[float, int]], max_m: int) -> list[int]:
        return [item_id for _, item_id in candidates[:max_m]]

    def insert(self, item: VectorItem, dist_fn: DistFn) -> None:
        item_id = item.id
        level = self._rand_level()
        self.graph[item_id] = HNSWNode(
            item=item,
            max_layer=level,
            nbrs=[[] for _ in range(level + 1)],
        )

        if self.entry_point == -1:
            self.entry_point = item_id
            self.top_layer = level
            return

        entry_point = self.entry_point
        for layer in range(self.top_layer, level, -1):
            if layer < len(self.graph[entry_point].nbrs):
                window = self._search_layer(item.emb, entry_point, 1, layer, dist_fn)
                if window:
                    entry_point = window[0][1]

        for layer in range(min(self.top_layer, level), -1, -1):
            window = self._search_layer(item.emb, entry_point, self.ef_build, layer, dist_fn)
            max_m = self.m0 if layer == 0 else self.m
            selected = self._select_neighbors(window, max_m)
            self.graph[item_id].nbrs[layer] = list(selected)

            for neighbor_id in selected:
                if neighbor_id not in self.graph:
                    continue
                neighbor = self.graph[neighbor_id]
                if len(neighbor.nbrs) <= layer:
                    neighbor.nbrs.extend([[] for _ in range(layer + 1 - len(neighbor.nbrs))])
                connections = neighbor.nbrs[layer]
                connections.append(item_id)
                if len(connections) > max_m:
                    distances = [
                        (dist_fn(neighbor.item.emb, self.graph[candidate].item.emb), candidate)
                        for candidate in connections
                        if candidate in self.graph
                    ]
                    distances.sort(key=lambda pair: pair[0])
                    neighbor.nbrs[layer] = [candidate for _, candidate in distances[:max_m]]

            if window:
                entry_point = window[0][1]

        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = item_id

    def knn(
        self, q: list[float], k: int, ef: int, dist_fn: DistFn
    ) -> list[tuple[float, int]]:
        if self.entry_point == -1:
            return []

        entry_point = self.entry_point
        for layer in range(self.top_layer, 0, -1):
            if layer < len(self.graph[entry_point].nbrs):
                window = self._search_layer(q, entry_point, 1, layer, dist_fn)
                if window:
                    entry_point = window[0][1]

        window = self._search_layer(q, entry_point, max(ef, k), 0, dist_fn)
        return window[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.graph:
            return

        for node in self.graph.values():
            for layer in node.nbrs:
                while item_id in layer:
                    layer.remove(item_id)

        if self.entry_point == item_id:
            self.entry_point = -1
            for node_id in self.graph:
                if node_id != item_id:
                    self.entry_point = node_id
                    break

        del self.graph[item_id]

    def get_info(self) -> dict:
        max_layer = max(self.top_layer + 1, 1)
        nodes_per_layer = [0 for _ in range(max_layer)]
        edges_per_layer = [0 for _ in range(max_layer)]
        nodes = []
        edges = []

        for item_id, node in sorted(self.graph.items()):
            nodes.append(
                {
                    "id": item_id,
                    "metadata": node.item.metadata,
                    "category": node.item.category,
                    "maxLyr": node.max_layer,
                }
            )
            for layer in range(min(node.max_layer, max_layer - 1) + 1):
                nodes_per_layer[layer] += 1
                if layer >= len(node.nbrs):
                    continue
                for neighbor_id in node.nbrs[layer]:
                    if item_id < neighbor_id:
                        edges_per_layer[layer] += 1
                        edges.append({"src": item_id, "dst": neighbor_id, "lyr": layer})

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        return len(self.graph)


class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.store: dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1

    def insert(self, metadata: str, category: str, emb: list[float], dist_fn: DistFn) -> int:
        with self.lock:
            item = VectorItem(self.next_id, metadata, category, emb)
            self.next_id += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list[float], k: int, metric: str, algo: str) -> dict:
        with self.lock:
            dist_fn = get_dist_fn(metric)
            start_ns = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dist_fn)
            else:
                raw = self.hnsw.knn(q, k, 50, dist_fn)
            latency_us = (time.perf_counter_ns() - start_ns) // 1000

            hits = []
            for distance, item_id in raw:
                item = self.store.get(item_id)
                if item is None:
                    continue
                hits.append(
                    {
                        "id": item.id,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": round(distance, 6),
                        "embedding": [round(value, 4) for value in item.emb],
                    }
                )

            return {
                "results": hits,
                "latencyUs": latency_us,
                "algo": algo,
                "metric": metric,
            }

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self.lock:
            dist_fn = get_dist_fn(metric)

            def measure(fn: Callable[[], None]) -> int:
                start_ns = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - start_ns) // 1000

            return {
                "bruteforceUs": measure(lambda: self.bf.knn(q, k, dist_fn)),
                "kdtreeUs": measure(lambda: self.kdt.knn(q, k, dist_fn)),
                "hnswUs": measure(lambda: self.hnsw.knn(q, k, 50, dist_fn)),
                "itemCount": len(self.store),
            }

    def all(self) -> list[VectorItem]:
        with self.lock:
            return [self.store[item_id] for item_id in sorted(self.store)]

    def hnsw_info(self) -> dict:
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)


class DocumentDB:
    def __init__(self) -> None:
        self.store: dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            proxy = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(proxy, cosine)
            self.bf.insert(proxy)
            return item.id

    def search(
        self, q: list[float], k: int, max_dist: float = 0.7
    ) -> list[tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []
            raw = (
                self.bf.knn(q, k, cosine)
                if len(self.store) < 10
                else self.hnsw.knn(q, k, 50, cosine)
            )
            hits = []
            for distance, item_id in raw:
                item = self.store.get(item_id)
                if item is not None and distance <= max_dist:
                    hits.append((distance, item))
            return hits

    def remove(self, item_id: int) -> bool:
        with self.lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self) -> list[DocItem]:
        with self.lock:
            return [self.store[item_id] for item_id in sorted(self.store)]

    def size(self) -> int:
        with self.lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self.host = host
        self.port = port
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2:3b"
    
    def _request_json(
        self, method: str, path: str, payload: dict | None = None, timeout: int = 5
    ) -> tuple[int | None, dict | list | None]:
        body = None
        headers: dict[str, str] = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        conn = HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            if not raw:
                return response.status, {}
            try:
                return response.status, json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return response.status, {}
        except OSError:
            return None, None
        finally:
            conn.close()

    def is_available(self) -> bool:
        status, _ = self._request_json("GET", "/api/tags", timeout=2)
        return status == 200

    def embed(self, text: str) -> list[float]:
        status, payload = self._request_json(
            "POST",
            "/api/embeddings",
            {"model": self.embed_model, "prompt": text},
            timeout=30,
        )
        if status != 200 or not isinstance(payload, dict):
            return []
        embedding = payload.get("embedding", [])
        if not isinstance(embedding, list):
            return []
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError):
            return []

    def generate(self, prompt: str) -> str:
        status, payload = self._request_json(
            "POST",
            "/api/generate",
            {"model": self.gen_model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        if status != 200 or not isinstance(payload, dict):
            return "ERROR: Ollama unavailable. Run: ollama serve"
        response = payload.get("response")
        return response if isinstance(response, str) else ""


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


DEMO_ITEMS: list[tuple[str, str, list[float]]] = [
    (
        "Linked List: nodes connected by pointers",
        "cs",
        [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06],
    ),
    (
        "Binary Search Tree: O(log n) search and insert",
        "cs",
        [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10],
    ),
    (
        "Dynamic Programming: memoization overlapping subproblems",
        "cs",
        [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07],
    ),
    (
        "Graph BFS and DFS: breadth and depth first traversal",
        "cs",
        [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07],
    ),
    (
        "Hash Table: O(1) lookup with collision chaining",
        "cs",
        [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09],
    ),
    (
        "Calculus: derivatives integrals and limits",
        "math",
        [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10],
    ),
    (
        "Linear Algebra: matrices eigenvalues eigenvectors",
        "math",
        [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09],
    ),
    (
        "Probability: distributions random variables Bayes theorem",
        "math",
        [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08],
    ),
    (
        "Number Theory: primes modular arithmetic RSA cryptography",
        "math",
        [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06],
    ),
    (
        "Combinatorics: permutations combinations generating functions",
        "math",
        [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07],
    ),
    (
        "Neapolitan Pizza: wood-fired dough San Marzano tomatoes",
        "food",
        [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07],
    ),
    (
        "Sushi: vinegared rice raw fish and nori rolls",
        "food",
        [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08],
    ),
    (
        "Ramen: noodle soup with chashu pork and soft-boiled eggs",
        "food",
        [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06],
    ),
    (
        "Tacos: corn tortillas with carnitas salsa and cilantro",
        "food",
        [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09],
    ),
    (
        "Croissant: laminated pastry with buttery flaky layers",
        "food",
        [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06],
    ),
    (
        "Basketball: fast-paced shooting dribbling slam dunks",
        "sports",
        [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72],
    ),
    (
        "Football: tackles touchdowns field goals and strategy",
        "sports",
        [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76],
    ),
    (
        "Tennis: racket volleys groundstrokes and Wimbledon serves",
        "sports",
        [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82],
    ),
    (
        "Chess: openings endgames tactics strategic board game",
        "sports",
        [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90],
    ),
    (
        "Swimming: butterfly freestyle backstroke Olympic competition",
        "sports",
        [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80],
    ),
]


def load_demo(db: VectorDB) -> None:
    dist_fn = get_dist_fn("cosine")
    for metadata, category, emb in DEMO_ITEMS:
        db.insert(metadata, category, emb, dist_fn)


class AppState:
    def __init__(self) -> None:
        self.db = VectorDB(DIMS)
        self.doc_db = DocumentDB()
        self.ollama = OllamaClient()
        load_demo(self.db)


APP_STATE = AppState()


def parse_vector(raw: str) -> list[float]:
    if not raw:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class MyAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MyAI/1.0"

    def _set_headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._set_headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_text(self, status: int, payload: str, content_type: str) -> None:
        body = payload.encode("utf-8")
        self._set_headers(status, content_type, len(body))
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = safe_int(self.headers.get("Content-Length"), 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def do_OPTIONS(self) -> None:
        self._set_headers(204, "text/plain; charset=utf-8", 0)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path in {"/", "/index.html"}:
            index_path = Path(__file__).with_name("index.html")
            if not index_path.exists():
                self._send_text(404, "index.html not found", "text/plain; charset=utf-8")
                return
            self._send_text(200, index_path.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            return

        if path == "/items":
            payload = [
                {
                    "id": item.id,
                    "metadata": item.metadata,
                    "category": item.category,
                    "embedding": [round(value, 4) for value in item.emb],
                }
                for item in APP_STATE.db.all()
            ]
            self._send_json(200, payload)
            return

        if path == "/search":
            q = parse_vector(params.get("v", [""])[0])
            if len(q) != DIMS:
                self._send_json(400, {"error": f"need {DIMS}D vector"})
                return

            k = safe_int(params.get("k", ["5"])[0], 5)
            metric = params.get("metric", ["cosine"])[0] or "cosine"
            algo = params.get("algo", ["hnsw"])[0] or "hnsw"
            self._send_json(200, APP_STATE.db.search(q, k, metric, algo))
            return

        if path == "/benchmark":
            q = parse_vector(params.get("v", [""])[0])
            if len(q) != DIMS:
                self._send_json(400, {"error": f"need {DIMS}D vector"})
                return

            k = safe_int(params.get("k", ["5"])[0], 5)
            metric = params.get("metric", ["cosine"])[0] or "cosine"
            self._send_json(200, APP_STATE.db.benchmark(q, k, metric))
            return

        if path == "/hnsw-info":
            self._send_json(200, APP_STATE.db.hnsw_info())
            return

        if path == "/status":
            available = APP_STATE.ollama.is_available()
            self._send_json(
                200,
                {
                    "ollamaAvailable": available,
                    "embedModel": APP_STATE.ollama.embed_model,
                    "genModel": APP_STATE.ollama.gen_model,
                    "docCount": APP_STATE.doc_db.size(),
                    "docDims": APP_STATE.doc_db.get_dims(),
                    "demoDims": DIMS,
                    "demoCount": APP_STATE.db.size(),
                },
            )
            return

        if path == "/stats":
            self._send_json(
                200,
                {
                    "count": APP_STATE.db.size(),
                    "dims": DIMS,
                    "algorithms": ["bruteforce", "kdtree", "hnsw"],
                    "metrics": ["euclidean", "cosine", "manhattan"],
                },
            )
            return

        if path == "/doc/list":
            docs = []
            for item in APP_STATE.doc_db.all():
                preview = item.text[:120] + ("..." if len(item.text) > 120 else "")
                docs.append(
                    {
                        "id": item.id,
                        "title": item.title,
                        "preview": preview,
                        "words": max(len(item.text.split()), 1),
                    }
                )
            self._send_json(200, docs)
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json()

        if path == "/insert":
            metadata = payload.get("metadata", "")
            category = payload.get("category", "")
            embedding = payload.get("embedding", [])
            if not isinstance(metadata, str) or not isinstance(embedding, list):
                self._send_json(400, {"error": "invalid body"})
                return

            try:
                emb = [float(value) for value in embedding]
            except (TypeError, ValueError):
                self._send_json(400, {"error": "invalid body"})
                return

            if len(emb) != DIMS:
                self._send_json(400, {"error": "invalid body"})
                return

            item_id = APP_STATE.db.insert(metadata, str(category), emb, get_dist_fn("cosine"))
            self._send_json(200, {"id": item_id})
            return

        if path == "/doc/insert":
            title = payload.get("title", "")
            text = payload.get("text", "")
            if not isinstance(title, str) or not isinstance(text, str) or not title or not text:
                self._send_json(400, {"error": "need title and text"})
                return

            chunks = chunk_text(text, 250, 30)
            ids: list[int] = []

            for index, chunk in enumerate(chunks):
                emb = APP_STATE.ollama.embed(chunk)
                if not emb:
                    self._send_json(
                        400,
                        {
                            "error": (
                                "Ollama unavailable. Install from https://ollama.com "
                                "then run: ollama pull nomic-embed-text && ollama pull llama3.2"
                            )
                        },
                    )
                    return
                chunk_title = (
                    f"{title} [{index + 1}/{len(chunks)}]" if len(chunks) > 1 else title
                )
                ids.append(APP_STATE.doc_db.insert(chunk_title, chunk, emb))

            self._send_json(
                200,
                {
                    "ids": ids,
                    "chunks": len(chunks),
                    "dims": APP_STATE.doc_db.get_dims(),
                },
            )
            return

        if path == "/doc/search":
            question = payload.get("question", "")
            k = safe_int(str(payload.get("k", 3)), 3)
            if not isinstance(question, str) or not question:
                self._send_json(400, {"error": "need question"})
                return

            q_emb = APP_STATE.ollama.embed(question)
            if not q_emb:
                self._send_json(400, {"error": "Ollama unavailable"})
                return

            hits = APP_STATE.doc_db.search(q_emb, k)
            self._send_json(
                200,
                {
                    "contexts": [
                        {
                            "id": item.id,
                            "title": item.title,
                            "distance": round(distance, 4),
                        }
                        for distance, item in hits
                    ]
                },
            )
            return

        if path == "/doc/ask":
            question = payload.get("question", "")
            k = safe_int(str(payload.get("k", 3)), 3)
            if not isinstance(question, str) or not question:
                self._send_json(400, {"error": "need question"})
                return

            q_emb = APP_STATE.ollama.embed(question)
            if not q_emb:
                self._send_json(400, {"error": "Ollama unavailable"})
                return

            hits = APP_STATE.doc_db.search(q_emb, k)
            context_lines = []
            for index, (_, item) in enumerate(hits, start=1):
                context_lines.append(f"[{index}] {item.title}:\n{item.text}\n")

            prompt = (
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it does not, use your own general knowledge. "
                "Do not mention the words context or provided text. "
                "Just answer naturally.\n\n"
                f"Context:\n{''.join(context_lines)}\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            answer = APP_STATE.ollama.generate(prompt)
            self._send_json(
                200,
                {
                    "answer": answer,
                    "model": APP_STATE.ollama.gen_model,
                    "contexts": [
                        {
                            "id": item.id,
                            "title": item.title,
                            "text": item.text,
                            "distance": round(distance, 4),
                        }
                        for distance, item in hits
                    ],
                    "docCount": APP_STATE.doc_db.size(),
                },
            )
            return

        self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/delete/"):
            try:
                item_id = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self._send_json(400, {"error": "invalid id"})
                return
            self._send_json(200, {"ok": APP_STATE.db.remove(item_id)})
            return

        if path.startswith("/doc/delete/"):
            try:
                item_id = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self._send_json(400, {"error": "invalid id"})
                return
            self._send_json(200, {"ok": APP_STATE.doc_db.remove(item_id)})
            return

        self._send_json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: object) -> None:
        message = fmt % args
        print(f"[http] {self.address_string()} - {message}")


def main() -> None:
    ollama_up = APP_STATE.ollama.is_available()
    print("=== MyAI ===")
    print(f"http://localhost:{PORT}")
    print(
        f"{APP_STATE.db.size()} demo vectors | {DIMS} dims | "
        "HNSW + KD-Tree + Brute Force"
    )
    print(
        "Ollama: "
        + (
            "ONLINE"
            if ollama_up
            else "OFFLINE (install from https://ollama.com)"
        )
    )
    if ollama_up:
        print(
            f"  embed model: {APP_STATE.ollama.embed_model}  "
            f"gen model: {APP_STATE.ollama.gen_model}"
        )

    server = ThreadingHTTPServer((HOST, PORT), MyAIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MyAI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
