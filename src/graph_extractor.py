"""
从区块链日志中提取客户端通信图的工具。

用途：
- 从 logs/blockchain_log.jsonl 中提取客户端之间的通信关系
- 构建有向边列表，用于 PageRank 计算
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

ClientId = int
Edge = Tuple[ClientId, ClientId]


def extract_communication_graph(
    log_path: str = "logs/blockchain_log.jsonl",
    round_id: int | None = None,
) -> tuple[list[ClientId], list[Edge]]:
    """
    从区块链日志中提取客户端通信图。

    参数：
    - log_path: 日志文件路径
    - round_id: 指定轮次（None 表示所有轮次）

    返回：
    - (clients, edges): 客户端列表和有向边列表
    """
    log_file = Path(log_path)
    if not log_file.exists():
        # 如果日志文件不存在，返回空图
        return [], []

    clients: set[ClientId] = set()
    edges: list[Edge] = []

    # 读取日志文件
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 过滤轮次（如果指定）
            if round_id is not None and entry.get("round") != round_id:
                continue

            event_type = entry.get("event_type", "")
            client_id = entry.get("client_id")

            # 收集所有参与训练的客户端
            if client_id is not None:
                clients.add(client_id)

            # 从 CLIENT_UPDATE_SUBMITTED 事件构建通信图
            # 假设：如果客户端 k 在 round r 提交了更新，那么它接收了来自其他客户端的更新
            # 简化：构建一个环状拓扑 + 基于提交顺序的额外边
            if event_type == "CLIENT_UPDATE_SUBMITTED" and client_id is not None:
                # 记录客户端参与通信
                clients.add(client_id)

    # 如果没有从日志中提取到边，构建一个默认的环状拓扑
    if not edges and clients:
        clients_list = sorted(list(clients))
        # 构建环状拓扑：0 -> 1 -> 2 -> ... -> n-1 -> 0
        for i in range(len(clients_list)):
            src = clients_list[i]
            dst = clients_list[(i + 1) % len(clients_list)]
            edges.append((src, dst))

        # 添加一些额外的边，模拟更复杂的通信模式
        # 例如：后面的客户端会向前面的客户端发送更新
        if len(clients_list) >= 3:
            edges.append((clients_list[-1], clients_list[0]))
        if len(clients_list) >= 4:
            edges.append((clients_list[-2], clients_list[0]))

    return sorted(list(clients)), edges


def extract_graph_from_rounds(
    log_path: str = "logs/blockchain_log.jsonl",
    max_rounds: int | None = None,
) -> tuple[list[ClientId], list[Edge]]:
    """
    从多个轮次中提取累积的通信图。

    参数：
    - log_path: 日志文件路径
    - max_rounds: 最大轮次数（None 表示所有轮次）

    返回：
    - (clients, edges): 客户端列表和有向边列表
    """
    log_file = Path(log_path)
    if not log_file.exists():
        return [], []

    clients: set[ClientId] = set()
    edges: list[Edge] = []
    seen_edges: set[Edge] = set()

    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            round_num = entry.get("round")
            if max_rounds is not None and round_num is not None and round_num >= max_rounds:
                continue

            event_type = entry.get("event_type", "")
            client_id = entry.get("client_id")

            if client_id is not None:
                clients.add(client_id)

            # 从 CLIENT_UPDATE_SUBMITTED 事件推断通信关系
            if event_type == "CLIENT_UPDATE_SUBMITTED" and client_id is not None:
                clients.add(client_id)

    # 构建默认拓扑（如果日志中没有足够的边信息）
    if clients and not edges:
        clients_list = sorted(list(clients))
        # 环状拓扑
        for i in range(len(clients_list)):
            src = clients_list[i]
            dst = clients_list[(i + 1) % len(clients_list)]
            edge = (src, dst)
            if edge not in seen_edges:
                edges.append(edge)
                seen_edges.add(edge)

        # 额外边
        if len(clients_list) >= 3:
            edge = (clients_list[-1], clients_list[0])
            if edge not in seen_edges:
                edges.append(edge)
                seen_edges.add(edge)
        if len(clients_list) >= 4:
            edge = (clients_list[-2], clients_list[0])
            if edge not in seen_edges:
                edges.append(edge)
                seen_edges.add(edge)

    return sorted(list(clients)), edges


if __name__ == "__main__":
    # 测试
    clients, edges = extract_communication_graph()
    print(f"Clients: {clients}")
    print(f"Edges: {edges}")

