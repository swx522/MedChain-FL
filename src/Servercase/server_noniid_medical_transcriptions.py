print("Start Training!!!!")

import psutil
import time
from collections import OrderedDict
import os
import random
import warnings
import flwr as fl
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, ClassLabel
from evaluate import load as load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import logging

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

def load_data_clients(i):
    """根据客户端索引 i 加载特定的数据切片"""
    raw_datasets = load_dataset("bhargavi909/Medical_Transcriptions_upsampled")
    
    if "unsupervised" in raw_datasets:
        del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    unique_labels = sorted(list(set(raw_datasets["train"]["medical_specialty"])))
    label_feature = ClassLabel(names=unique_labels)

    def preprocess_function(examples):
        tokenized = tokenizer(examples["description"], truncation=True, max_length=128)
        new_labels = []
        for l in examples["medical_specialty"]:
            if isinstance(l, int):
                new_labels.append(l)
            else:
                new_labels.append(label_feature.str2int(str(l)))
        
        tokenized["labels"] = new_labels
        return tokenized

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(int(500*i), int(int(500*i)+400)))
    tokenized_datasets["test"] = tokenized_datasets["test"].select(range(0, 400)) # 测试集保持一致

    tokenized_datasets = tokenized_datasets.remove_columns(["description", "medical_specialty", "Unnamed: 0"])
    
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
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


"""## Federating the example

The idea behind Federated Learning is to train a model between multiple clients and a server without having to share any data. This is done by letting each client train the model locally on its data and send its parameters back to the server, which then aggregates all the clients’ parameters together using a predefined strategy. This process is made very simple by using the [Flower](https://github.com/adap/flower) framework. If you want a more complete overview, be sure to check out this guide: [What is Federated Learning?](https://flower.dev/docs/tutorial/Flower-0-What-is-FL.html)

### Creating the IMDBClient

To federate our example to multiple clients, we first need to write our Flower client class (inheriting from `flwr.client.NumPyClient`). This is very easy, as our model is a standard `PyTorch` model:
"""

class IMDBClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, epochs=1)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy), "loss": float(loss)}

"""### Generating the clients"""
def client_fn(cid):
    trainloader, testloader = load_data_clients(int(cid))
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=40, ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    return IMDBClient(net, trainloader, testloader)

"""## Starting the simulation"""

def weighted_average(metrics):
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  losses = [num_examples * m["loss"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0},
    ray_init_args={"log_to_driver": False, "num_cpus": 1, "num_gpus": 0}
)

"""Note that this is a very basic example, and a lot can be added or modified, it was just to showcase how simply we could federate a Hugging Face workflow using Flower. The number of clients and the data samples are intentionally very small in order to quickly run inside Colab, but keep in mind that everything can be tweaked and extended."""
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
