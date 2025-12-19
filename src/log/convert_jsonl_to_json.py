import json

JSONL_PATH="logs/blockchain_log.jsonl"
JSON_PATH="logs/blockchain_log.json"

def jsonl_to_json() -> None:
    data = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {i} 行不是合法 JSON：{e}\n内容：{line[:200]}") from e

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"OK: {JSONL_PATH} -> {JSON_PATH}  (共 {len(data)} 条)")

if __name__ == "__main__":
    jsonl_to_json()