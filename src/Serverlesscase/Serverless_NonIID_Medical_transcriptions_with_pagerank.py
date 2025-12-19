print("Start Training with PageRank!!!!")

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入 src 模块
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import psutil
import time
from collections import OrderedDict
import os
import random
import warnings
import flwr as fl
import torch
import copy
from torch.utils.data import DataLoader
from datasets import load_dataset, ClassLabel
from evaluate import load as load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import logging
from src.log.logger import log_event
from src.pagerank_utils import compute_pagerank_scores, sort_scores_desc
from src.graph_extractor import extract_graph_from_rounds

"""Next we will set some global variables and disable some of the logging to clear out our output."""

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "dmis-lab/biobert-v1.1"
NUM_ROUNDS = 3
NUM_CLIENTS = 8
PAGERANK_THRESHOLD = 0.3  # 剔除阈值：PageRank分数 < 0.3 的客户端将被剔除

before_communication_cpu_percent = psutil.cpu_percent()
current_process = psutil.Process()

memory_info_after = current_process.memory_info()
start = time.time()

# 训练开始日志（用于答辩截图/追踪配置）
log_event(
    event_type="TRAIN_START",
    message="训练开始（Serverless_NonIID_Medical_transcriptions_with_pagerank.py）",
    details={
        "checkpoint": CHECKPOINT,
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "device": str(DEVICE),
        "pagerank_threshold": PAGERANK_THRESHOLD,
        "use_pagerank": True,
    },
)

raw_temp = load_dataset("bhargavi909/Medical_Transcriptions_upsampled", split="train")
unique_labels = sorted(list(set([str(l) for l in raw_temp["medical_specialty"]])))
label_feature = ClassLabel(names=unique_labels)
NUM_LABELS = len(unique_labels)

def load_data(client_id):
    """Load Medical data (training and eval)"""
    raw_datasets = load_dataset("bhargavi909/Medical_Transcriptions_upsampled")
    if "unsupervised" in raw_datasets:
        del raw_datasets["unsupervised"]
    
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples["description"], padding=True, truncation=True, max_length=128)
        new_labels = []
        for l in examples["medical_specialty"]:
            if isinstance(l, int): new_labels.append(l)
            else: new_labels.append(label_feature.str2int(str(l)))
        tokenized["labels"] = new_labels
        return tokenized

    start_idx = client_id * 200
    end_idx = start_idx + 200

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(start_idx, end_idx))
    tokenized_datasets["test"] = tokenized_datasets["test"].select(range(0, 100))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader

def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    if len(testloader.dataset) == 0: return 0, 0
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

class IMDBClient:
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def train_model(self):
        train(self.net, self.trainloader, epochs=1)

    def evaluate_model(self):
        loss, accuracy = test(self.net, self.testloader)
        return loss, accuracy

global_model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
).to(DEVICE)

def evaluate_global_model(model, testloader):
    """Evaluate the global model on the test dataset."""
    loss, accuracy = test(model, testloader)
    return accuracy

global_accuracies = []
active_clients = set(range(NUM_CLIENTS))  # 活跃客户端集合

for round_num in range(NUM_ROUNDS):
    print(f"--- Round {round_num + 1} ---")
    print(f"Active clients: {sorted(active_clients)}")

    # 每轮开始日志
    log_event(
        event_type="ROUND_START",
        round_id=round_num,
        message=f"Round {round_num + 1} start",
        details={"num_clients": len(active_clients), "active_clients": sorted(list(active_clients))},
    )

    aggregated_params = []
    client_params_dict = {}  # 存储每个客户端的参数

    for k in active_clients:
        # 客户端本地训练开始
        log_event(
            event_type="CLIENT_TRAIN_START",
            round_id=round_num,
            client_id=k,
            message=f"Client {k} start local training",
        )

        trainloader, testloader = load_data(k)
        client_model = copy.deepcopy(global_model)
        
        client = IMDBClient(client_model, trainloader, testloader)
        client.train_model()

        # 客户端本地训练结束
        log_event(
            event_type="CLIENT_TRAIN_END",
            round_id=round_num,
            client_id=k,
            message=f"Client {k} finish local training",
        )

        client_params = client.get_parameters(config={})

        # 模型提交（客户端提交更新）
        log_event(
            event_type="CLIENT_UPDATE_SUBMITTED",
            round_id=round_num,
            client_id=k,
            message=f"Client {k} submitted model update",
            details={"num_tensors": len(client_params)},
        )

        loss, accuracy = client.evaluate_model()
        print(f"Client {k} Acc: {accuracy:.4f}")

        # 客户端本地评估结果
        log_event(
            event_type="CLIENT_LOCAL_EVAL",
            round_id=round_num,
            client_id=k,
            message=f"Client {k} local evaluation done",
            details={"loss": float(loss), "accuracy": float(accuracy)},
        )

        client_params_dict[k] = client_params
        aggregated_params.append(client_params)

    # 在聚合之前，计算 PageRank 并剔除低分客户端
    # 从第二轮开始应用 PageRank（第一轮需要先有日志）
    if round_num > 0:
        print("\n=== Computing PageRank scores ===")
        
        # 从日志中提取通信图
        clients_from_log, edges_from_log = extract_graph_from_rounds(max_rounds=round_num + 1)
        
        # 重要：只对当前活跃的客户端计算 PageRank（过滤掉已被剔除的客户端）
        # 如果日志中的客户端包含已被剔除的，需要过滤
        active_clients_set = set(active_clients)
        clients = sorted([c for c in clients_from_log if c in active_clients_set])
        
        # 如果没有从日志中提取到足够的活跃客户端，直接使用当前活跃的客户端
        if not clients or len(clients) < len(active_clients):
            clients = sorted(list(active_clients))
        
        # 构建基于活跃客户端的图结构
        # 过滤边：只保留连接活跃客户端的边
        edges = []
        if edges_from_log:
            edges = [(src, dst) for src, dst in edges_from_log 
                    if src in active_clients_set and dst in active_clients_set]
        
        # 如果没有有效的边，构建默认的环状拓扑（基于活跃客户端）
        if not edges and clients:
            for i in range(len(clients)):
                src = clients[i]
                dst = clients[(i + 1) % len(clients)]
                edges.append((src, dst))
            # 添加额外边
            if len(clients) >= 3:
                edges.append((clients[-1], clients[0]))
            if len(clients) >= 4:
                edges.append((clients[-2], clients[0]))
        
        if clients and edges:
            # 计算 PageRank 分数
            # PageRank 返回的原始分数：所有客户端分数之和为 1，但分数分布可能不均匀
            # 例如：8个客户端时，分数可能在 [0.07, 0.14] 之间，平均值约为 0.125
            pagerank_scores_raw = compute_pagerank_scores(clients, edges)
            
            # ===== PageRank 分数归一化 =====
            # 目的：将原始 PageRank 分数映射到 [0, 1] 区间，使得阈值判断更加公平和稳定
            # 
            # 原因：
            # 1. 原始 PageRank 分数的绝对值范围会随客户端数量变化而变化
            #    例如：8个客户端时分数在 [0.07, 0.14]，但100个客户端时可能在 [0.005, 0.02]
            # 2. 使用固定阈值（如0.3）在客户端数量不同时会产生不一致的筛选效果
            # 3. 归一化后，分数范围始终是 [0, 1]，使得阈值具有统一的语义
            #    0.3 表示"相对排名在前70%"，而不是"绝对分数大于0.3"
            #
            # 归一化公式：pr_norm = (pr - min) / (max - min)
            # - 将最低分映射到 0，最高分映射到 1
            # - 其他分数按比例线性映射到 [0, 1] 区间
            #
            # 注意：如果所有客户端分数相同（max == min），则归一化后所有分数都为 0.5
            #       这种情况下不会剔除任何客户端（因为 0.5 > 0.3）
            if pagerank_scores_raw:
                scores_list = list(pagerank_scores_raw.values())
                min_score = min(scores_list)
                max_score = max(scores_list)
                
                if max_score > min_score:
                    # 正常情况：有分数差异，进行归一化
                    pagerank_scores = {
                        cid: (score - min_score) / (max_score - min_score)
                        for cid, score in pagerank_scores_raw.items()
                    }
                    print(f"PageRank normalization: [{min_score:.4f}, {max_score:.4f}] -> [0.0, 1.0]")
                else:
                    # 特殊情况：所有分数相同，归一化后都设为 0.5（避免除以0）
                    pagerank_scores = {cid: 0.5 for cid in pagerank_scores_raw.keys()}
                    print(f"PageRank normalization: all scores equal ({min_score:.4f}), normalized to 0.5")
            else:
                pagerank_scores = {}
            
            # 保存 PageRank 分数到文件（保存归一化后的分数）
            root_dir = Path(__file__).resolve().parent.parent.parent
            scores_path = root_dir / "pagerank_scores.txt"
            with scores_path.open("w", encoding="utf-8") as f:
                f.write(f"# Round {round_num + 1} - PageRank scores (normalized)\n")
                f.write("# client_id\tpagerank_score_raw\tpagerank_score_normalized\n")
                for cid, score_raw in sort_scores_desc(pagerank_scores_raw):
                    score_norm = pagerank_scores.get(cid, 0.0)
                    f.write(f"{cid}\t{score_raw:.6f}\t{score_norm:.6f}\n")
            
            print(f"PageRank scores saved to: {scores_path}")
            print("PageRank scores (normalized):")
            for cid, score in sort_scores_desc(pagerank_scores):
                score_raw = pagerank_scores_raw.get(cid, 0.0)
                print(f"  client {cid}: {score:.4f} (raw: {score_raw:.4f})")
            
            # 标记并剔除低分客户端（使用归一化后的分数与阈值 0.3 比较）
            # 归一化后的分数在 [0, 1] 区间，0.3 表示相对排名在前70%的客户端
            removed_clients = []
            for client_id in list(active_clients):
                score_norm = pagerank_scores.get(client_id, 0.0)
                if score_norm < PAGERANK_THRESHOLD:
                    removed_clients.append(client_id)
                    active_clients.remove(client_id)
                    # 输出剔除日志（符合要求的格式）
                    score_raw = pagerank_scores_raw.get(client_id, 0.0)
                    print(f"⚠️  client {client_id} removed (normalized PageRank: {score_norm:.4f} < {PAGERANK_THRESHOLD}, raw: {score_raw:.4f})")
                    
                    # 记录剔除日志到区块链日志文件
                    log_event(
                        event_type="CLIENT_REMOVED",
                        round_id=round_num,
                        client_id=client_id,
                        message=f"client {client_id} removed",
                        details={
                            "pagerank_score_raw": float(score_raw),
                            "pagerank_score_normalized": float(score_norm),
                            "threshold": PAGERANK_THRESHOLD,
                            "reason": "low normalized PageRank score",
                        },
                    )
            
            # 从聚合参数中移除被剔除的客户端
            if removed_clients:
                aggregated_params = [
                    client_params_dict[k] 
                    for k in active_clients 
                    if k in client_params_dict
                ]
                print(f"Removed clients: {removed_clients}")
                print(f"Remaining active clients: {sorted(active_clients)}")
        else:
            print("Warning: Could not extract communication graph from logs")

    # Averaging the parameters
    if aggregated_params:
        avg_params = [sum(param) / len(param) for param in zip(*aggregated_params)]
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(global_model.state_dict().keys(), avg_params)})
        global_model.load_state_dict(state_dict)

        # 本轮聚合完成
        log_event(
            event_type="ROUND_AGGREGATED",
            round_id=round_num,
            message=f"Round {round_num + 1} aggregated (FedAvg)",
            details={"num_submitted_clients": len(aggregated_params)},
        )

        # Evaluate the global model
        trainloader, testloader = load_data(0)  # 使用第一个客户端的数据进行评估
        global_accuracy = evaluate_global_model(global_model, testloader)
        global_accuracies.append(global_accuracy)
        print(f"Global Model Accuracy: {global_accuracy * 100:.2f}%")

        # 全局评估结果
        log_event(
            event_type="GLOBAL_EVAL",
            round_id=round_num,
            message=f"Round {round_num + 1} global evaluation done",
            details={"global_accuracy": float(global_accuracy)},
        )

        global_model.save_pretrained('./medical_biobert_pagerank')

        # 模型保存事件
        log_event(
            event_type="GLOBAL_MODEL_SAVED",
            round_id=round_num,
            message="Global model saved",
            details={"path": "./medical_biobert_pagerank"},
        )

after_communication_cpu_percent = psutil.cpu_percent()
current_process = psutil.Process()

memory_info_before = current_process.memory_info()

# Calculate the communication overhead
cpu_overhead = after_communication_cpu_percent - before_communication_cpu_percent
memory_overhead =(memory_info_after.rss - memory_info_before.rss) / (1024 ** 3)  # Convert bytes to GB
end = time.time()

print(f"\n=== Training Summary ===")
print(f"CPU Overhead: {cpu_overhead}%")
print(f"Memory Usage: {memory_overhead:.2f} GB")
print(f"Latency: {(end-start)/60:.2f} min")
print("Global accuracies:")
for i, acc in enumerate(global_accuracies):
    print(f"  Round {i+1}: {acc * 100:.2f}%")

# 训练结束日志
log_event(
    event_type="TRAIN_END",
    message="训练结束（资源与精度汇总）- with PageRank",
    details={
        "cpu_overhead_percent": float(cpu_overhead),
        "memory_overhead_gb": float(memory_overhead),
        "latency_min": float((end-start)/60),
        "global_accuracies": [float(x) for x in global_accuracies],
        "final_active_clients": sorted(list(active_clients)),
    },
)

