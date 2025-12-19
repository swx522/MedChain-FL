"""
一个“区块链风格”的追加式事件日志器（Append-only Logger）

输出文件：blockchain_log.jsonl
- 采用 JSON Lines 格式：每一行是一个 JSON 对象（而不是一个大 JSON 数组）
- 这样便于“不断追加写入”，并且方便后续按行读取、过滤、统计、截图展示
"""

import json
import os
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager
import hashlib

LOG_PATH="logs/blockchain_log.jsonl"

def _canonical_json(obj: Dict[str, Any]) -> str:
    """
    稳定序列化：排序 key + 固定分隔符
    目的：保证同一条日志对象序列化结果稳定，从而 hash 可复现
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(s: str) -> str:
    """计算 SHA-256，用于生成链式哈希"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@contextmanager
def _file_lock(f):
    """
    尽量做一个跨平台文件锁（best-effort）：
    - Linux/macOS：fcntl.flock
    - Windows：msvcrt.locking
    目的：多进程/多客户端同时写日志时，降低“行交错/断链”的概率。
    如果锁失败，也不会让训练直接崩（继续无锁写）。
    """
    try:
        if os.name == "nt":
            import msvcrt  # Windows

            # 简化处理：锁定文件开头 1 字节（足够实现互斥）
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # Unix

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        # 锁不可用时，继续运行（不影响训练主流程）
        yield


"""
写入一条事件日志（追加写入、一行一个 JSON）。

参数含义：
- event_type：事件类型（机器可读），如 CLIENT_UPDATE_SUBMITTED / CLIENT_REMOVED
- round_id：第几轮（可选）
- client_id：哪个客户端（可选）
- message：人类可读摘要（用于答辩截图）
- details：结构化细节（dict），例如 pagerank_score、threshold、accuracy、loss 等
- log_path：输出文件路径（默认 blockchain_log.json）
- chain_hash：是否启用“链式哈希”（prev_hash -> curr_hash），更贴合区块链主题

返回值：
- 返回写入的 entry（便于调试或单元测试）
"""
def log_event(
    *,
    event_type: str,
    round_id: Optional[int] = None,
    client_id: Optional[int] = None,
    message: str = "",
    details: Optional[Dict[str, Any]] = None,
    chain_hash: bool = True,
)-> Dict[str, Any]:
    if details is None:
        details = {}

    # 统一字段结构：方便后续统计与展示
    entry: Dict[str, Any] = {
        "ts_ms": int(time.time() * 1000),  # 毫秒级时间戳
        "event_type": event_type,
        "round": round_id,
        "client_id": client_id,
        "message": message,
        "details": details,
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(LOG_PATH)or ".", exist_ok=True)

    # 用 a+：既可以读到最后一行（拿 prev_hash），也可以追加写入
    # a+:
    # append:写入永远在文件末尾追加，不会覆盖之前内容(追加本条日志)
    # read:允许读(为了读到上一条日志的hash->prev_hash)
    with open(LOG_PATH, "a+", encoding="utf-8") as f:
        with _file_lock(f):
            prev_hash = "GENESIS"  # 第一条日志的前置哈希(默认值)

            # 读取上一条日志的 curr_hash，作为本条的 prev_hash
            if chain_hash:
                # 把文件指针移动到文件末尾 
                f.seek(0, os.SEEK_END)
                # 非空文件
                if f.tell() > 0:
                    # 从文件末尾向前扫描，找到最后一个换行符，读取最后一行
                    pos = f.tell() - 1
                    while pos > 0:
                        f.seek(pos, os.SEEK_SET)
                        if f.read(1) == "\n":
                            break
                        pos -= 1

                    last_line = f.readline().strip()
                    if last_line:
                        try:
                            last_obj = json.loads(last_line)
                            prev_hash = last_obj.get("curr_hash", "GENESIS")
                        except Exception:
                            prev_hash = "GENESIS"

                entry["prev_hash"] = prev_hash

                # 计算本条 curr_hash：对除 curr_hash 外所有字段做稳定序列化后 hash
                payload = dict(entry)
                payload.pop("curr_hash", None)
                payload_str = _canonical_json(payload)
                entry["curr_hash"] = _sha256(payload_str)

            # 追加写入一行 JSON
            f.write(_canonical_json(entry) + "\n")
            f.flush()

if __name__ == "__main__":
    log_event(event_type="TEST_EVENT", message="测试写入", details={"x": 1})
    print("OK, wrote one line to logs/blockchain_log.json")
