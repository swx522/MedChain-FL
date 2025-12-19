import psutil
import time
from collections import OrderedDict
import os
import random
import warnings
import copy

import flwr as fl
import torch

from torch.utils.data import DataLoader

from datasets import load_dataset
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
CHECKPOINT = "albert-base-v2"  # transformer model checkpoint
NUM_CLIENTS = 8
NUM_ROUNDS = 3

"""## Standard Hugging Face workflow

### Handling the data

To fetch the IMDB dataset, we will use Hugging Face's `datasets` library. We then need to tokenize the data and create `PyTorch` dataloaders, this is all done in the `load_data` function:
"""
before_communication_cpu_percent = psutil.cpu_percent()
current_process = psutil.Process()

memory_info_after = current_process.memory_info()
start = time.time()

def load_data(client_id):
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)

    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    start_idx = client_id * 100
    end_idx = start_idx + 100
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(start_idx, end_idx))
    tokenized_datasets["test"] = tokenized_datasets["test"].select(range(0, 100))

    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

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

"""### Training and testing the model"""

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

"""### Creating the IMDBClient"""

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
    CHECKPOINT, num_labels=2
).to(DEVICE)

def evaluate_global_model(model, testloader):
    """Evaluate the global model on the test dataset."""
    loss, accuracy = test(model, testloader)
    return accuracy

global_accuracies = []

for round_num in range(NUM_ROUNDS):
    print(f"--- Round {round_num + 1} ---")
    aggregated_params = []
    
    for k in range(NUM_CLIENTS):
        trainloader, testloader = load_data(k)
        
        client_model = copy.deepcopy(global_model)
        
        client = IMDBClient(client_model, trainloader, testloader)
        client.train_model()
        client_params = client.get_parameters(config={})
        loss, accuracy = client.evaluate_model()
        print(f"Client {k} Acc: {accuracy:.4f}")
        aggregated_params.append(client_params)

    # Averaging the parameters
    avg_params = [sum(param) / len(param) for param in zip(*aggregated_params)]
    
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(global_model.state_dict().keys(), avg_params)})
    global_model.load_state_dict(state_dict)

    # Evaluate the global model
    print(global_model)
    trainloader, testloader = load_data()
    global_accuracy = evaluate_global_model(global_model, testloader)
    global_accuracies.append(global_accuracy)
    print(f"Global Model Accuracy: {global_accuracy * 100:.2f}%")
    global_model.save_pretrained('./my_albert_model2')
# Get the file size in GB
    def get_dir_size(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    dir_size = get_dir_size('./my_albert_model2') / (1024 * 1024 * 1024)  # Size in GB
    print("Model size in GB")
    print(dir_size)

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
