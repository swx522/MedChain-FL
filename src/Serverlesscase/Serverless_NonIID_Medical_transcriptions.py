print("Start Training!!!!")

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




before_communication_cpu_percent = psutil.cpu_percent()
current_process = psutil.Process()

memory_info_after = current_process.memory_info()
start = time.time()

# 训练开始日志（用于答辩截图/追踪配置）
log_event(
    event_type="TRAIN_START",
    message="训练开始（Serverless_NonIID_Medical_transcriptions.py）",
    details={
        "checkpoint": CHECKPOINT,
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "device": str(DEVICE),
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

"""### Training and testing the model
Once we have a way of creating our trainloader and testloader, we can take care of the training and testing. This is very similar to any `PyTorch` training or testing loop:
"""
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

"""### Creating the model itself

To create the model itself, we will just load the pre-trained alBERT model using Hugging Face’s `AutoModelForSequenceClassification` :
"""

"""## Federating the example

The idea behind Federated Learning is to train a model between multiple clients and a server without having to share any data. This is done by letting each client train the model locally on its data and send its parameters back to the server, which then aggregates all the clients’ parameters together using a predefined strategy. This process is made very simple by using the [Flower](https://github.com/adap/flower) framework. If you want a more complete overview, be sure to check out this guide: [What is Federated Learning?](https://flower.dev/docs/tutorial/Flower-0-What-is-FL.html)

### Creating the IMDBClient

To federate our example to multiple clients, we first need to write our Flower client class (inheriting from `flwr.client.NumPyClient`). This is very easy, as our model is a standard `PyTorch` model:
"""

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

"""## Starting the simulation"""

global_model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
).to(DEVICE)

def evaluate_global_model(model, testloader):
    """Evaluate the global model on the test dataset."""
    loss, accuracy = test(model, testloader)
    return accuracy

global_accuracies = []
for round_num in range(NUM_ROUNDS):
    print(f"--- Round {round_num + 1} ---")

    # 每轮开始日志
    log_event(
        event_type="ROUND_START",
        round_id=round_num,
        message=f"Round {round_num + 1} start",
        details={"num_clients": NUM_CLIENTS},
    )

    aggregated_params = []
    for k in range(NUM_CLIENTS):

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
            # 注意：不要把参数数组写进日志（太大），只写元信息
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

            aggregated_params.append(client_params)

    # Averaging the parameters
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
    # print(global_model)
    trainloader, testloader = load_data()
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

    global_model.save_pretrained('./medical_biobert')

    # 模型保存事件
    log_event(
        event_type="GLOBAL_MODEL_SAVED",
        round_id=round_num,
        message="Global model saved",
        details={"path": "./medical_biobert"},
    )

# Get the file size in GB
    def get_dir_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    dir_size = get_dir_size('./medical_biobert') / (1024 * 1024 * 1024)  # Size in GB
    print("Model size in GB")
    print(dir_size)

    # 模型目录大小（用于汇报）
    log_event(
        event_type="GLOBAL_MODEL_SIZE",
        round_id=round_num,
        message="Model directory size computed",
        details={"dir_size_gb": float(dir_size)},
    )

after_communication_cpu_percent = psutil.cpu_percent()
current_process = psutil.Process()

memory_info_before = current_process.memory_info()

# Calculate the communication overhead
cpu_overhead = after_communication_cpu_percent - before_communication_cpu_percent
memory_overhead =(memory_info_after.rss - memory_info_before.rss) / (1024 ** 3)  # Convert bytes to GB
end = time.time()

print(f"CPU Overhead: {cpu_overhead}%")
print(f"Memory Usage: {memory_overhead:.2f} GB")
print(f"Latency: {(end-start)/60} min")
print("global accuracies")
print(global_accuracies)

# 训练结束日志
log_event(
    event_type="TRAIN_END",
    message="训练结束（资源与精度汇总）",
    details={
        "cpu_overhead_percent": float(cpu_overhead),
        "memory_overhead_gb": float(memory_overhead),
        "latency_min": float((end-start)/60),
        "global_accuracies": [float(x) for x in global_accuracies],
    },
)
