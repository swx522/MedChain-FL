# Blockchain Logger 使用说明（logger.py + convert_jsonl_to_json.py）

## 1. 文件概览

### 1.1 `logger.py`

一个“区块链风格”的**追加式事件日志器**（append-only），默认输出为 **JSON Lines（.jsonl）**：

* 输出文件：`logs/blockchain_log.jsonl`
* 每一行是一个 JSON 对象（不是一个大数组）
* 额外包含 `prev_hash` / `curr_hash`，形成链式哈希，便于追溯、避免篡改 

### 1.2 `convert_jsonl_to_json.py`

将 `logs/blockchain_log.jsonl`（JSON Lines）转换为**严格 JSON 文件（数组格式）**：

* 输入：`logs/blockchain_log.jsonl`
* 输出：`logs/blockchain_log.json`
* 生成的 `.json` 文件可被 VS Code 作为标准 JSON 正常打开（不会“不是合法 JSON”）

---

## 2. 依赖与目录要求

* Python 3.8+（推荐）
* 无需额外第三方库（只用标准库） 
* 确保项目目录下存在 `logs/` 文件夹（没有也没关系，`logger.py` 会自动创建）

---

## 3. logger.py 的使用方法

### 3.1 在代码中记录事件（推荐）

在你要记录事件的位置加入：

```python
from src.log.logger import log_event

log_event(
    event_type="CLIENT_UPDATE_SUBMITTED",
    round_id=0,
    client_id=3,
    message="客户端提交模型更新",
    details={"num_tensors": 201, "note": "不要把大参数数组写进日志"},
)
```

参数说明（核心几个）：

* `event_type`：事件类型（机器可读），例如 `CLIENT_UPDATE_SUBMITTED` / `CLIENT_REMOVED`
* `round_id`：轮次（可选）
* `client_id`：客户端编号（可选）
* `message`：人类可读摘要（用于答辩截图）
* `details`：结构化细节（dict），例如 accuracy、loss、pagerank_score 等 

> 注意：`log_event` 目前是 **关键字-only**（必须写 `event_type="..."`），不能写 `log_event("XXX")`。

---

### 3.2 直接运行 logger.py 做自测

在项目根目录执行：

```bash
python logger.py
```

它会写入一条测试事件 `TEST_EVENT` 到日志中，用于验证环境与路径是否正确。

日志文件位置：

* `logs/blockchain_log.jsonl` 

---

## 4. 为什么日志文件是 .jsonl 而不是 .json？

`logger.py` 使用 **JSON Lines**（一行一个 JSON 对象），原因：

* 训练过程中日志不断产生，JSON Lines 可以**直接追加写入**，无需把整个文件读出来再写回
* 更适合流式记录、调试、过滤与截图展示
* 每条日志有 `prev_hash` / `curr_hash`，形成链式结构，更贴合“区块链日志”主题 

---

## 5. 将 JSON Lines 转成标准 JSON（数组）用于提交/查看

VS Code 报错 “不是合法 JSON” 属于正常现象：因为 `.jsonl` 不是单个 JSON 值，而是多行 JSON。

如果你需要一个严格 JSON 文件（数组格式 `[...]`），执行：

```bash
python convert_jsonl_to_json.py
```

执行后会生成：

* `logs/blockchain_log.json`（标准 JSON 数组）

---

## 6. 常见问题（FAQ）

### Q1：为什么不直接写成 JSON 数组？

因为写成数组会导致每次追加都要“读全量文件 → append → 重写全量文件”，开销更大；JSON Lines 天生适合追加写。

### Q2：日志里不要写什么？

不要把模型参数（巨大的数组）直接写进 `details`，文件会非常大、转换也会很慢。建议只写：

* 张量数量 / 轮次 / client_id / 阈值 / 指标（accuracy/loss）等。

---
