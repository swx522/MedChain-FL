"""
简化版 PageRank 评分工具。

目标（对应 12.19 的 ddl）：
- 提供一个独立的函数，给定客户端列表和它们之间的有向边，计算每个客户端的 PageRank 得分。
- 后续可以很容易地集成到训练循环中，用于异常客户端检测和剔除。

说明：
- 这里不关心“图是怎么来的”，只负责在已经有 (clients, edges) 的情况下计算分数。
- 使用 networkx.pagerank 实现。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx


ClientId = int
Edge = Tuple[ClientId, ClientId]


def build_client_graph(clients: Iterable[ClientId], edges: Iterable[Edge]) -> nx.DiGraph:
    """
    根据客户端 ID 列表和有向边构建一个有向图。

    参数：
    - clients: 客户端 ID，可迭代，例如 [0, 1, 2, 3]
    - edges:   有向边列表，例如 [(0, 1), (1, 2)] 表示 0 -> 1, 1 -> 2

    返回：
    - networkx.DiGraph 对象
    """
    g = nx.DiGraph()
    g.add_nodes_from(clients)
    g.add_edges_from(edges)
    return g


def compute_pagerank_scores(
    clients: Iterable[ClientId],
    edges: Iterable[Edge],
    alpha: float = 0.85,
) -> Dict[ClientId, float]:
    """
    计算每个客户端的 PageRank 得分。

    参数：
    - clients: 客户端 ID 列表
    - edges:   有向边 (src, dst) 列表
    - alpha:   PageRank 的阻尼系数，一般取 0.85

    返回：
    - {client_id: score} 的字典，score 为 [0, 1] 之间的浮点数，所有分数之和约为 1
    """
    graph = build_client_graph(clients, edges)

    if graph.number_of_nodes() == 0:
        return {}

    # 使用 networkx 提供的 PageRank 实现
    scores = nx.pagerank(graph, alpha=alpha)

    # 确保只返回整数 client_id 的键（networkx 可能会保留任意 hashable 节点）
    return {int(node): float(score) for node, score in scores.items()}


def sort_scores_desc(scores: Dict[ClientId, float]) -> List[Tuple[ClientId, float]]:
    """
    将 {client_id: score} 排序后返回一个列表，按 score 从高到低。
    方便后续打印或做阈值筛选。
    """
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


if __name__ == "__main__":
    # 一个简单的自测示例：8 个客户端，构造一个“环 + 若干额外边”的拓扑
    num_clients = 8
    clients = list(range(num_clients))

    edges: List[Edge] = []

    # 环结构：0 -> 1 -> 2 -> ... -> 7 -> 0
    for i in range(num_clients):
        src = i
        dst = (i + 1) % num_clients
        edges.append((src, dst))

    # 额外给前 2 个客户端一些“优势”边，模拟更重要的节点
    edges.extend(
        [
            (2, 0),
            (3, 0),
            (4, 1),
            (5, 1),
        ]
    )

    scores = compute_pagerank_scores(clients, edges)
    print("Demo PageRank scores:")
    for cid, score in sort_scores_desc(scores):
        print(f"client {cid}: {score:.4f}")


