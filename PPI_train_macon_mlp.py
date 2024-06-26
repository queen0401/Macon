import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import math
from tqdm import tqdm
import matplotlib.pyplot as plt 
from math import sqrt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
    
class Embedding_CNN(nn.Module):
    def __init__(self, input_size):
        super(Embedding_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64, 1024)  # 根据你的数据调整
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(x.shape) #32, 1024, 513
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape) #32, 256, 256
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) #32, 128, 128
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape) #32, 64, 64
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        return x
    
class SequenceModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SequenceModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1_emb = nn.Linear(1024 * 3, 1024)
        self.fc1 = nn.Linear(64 * 320, 1024)
        self.fc2 = nn.Linear(1024 * 2, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.norm_x = nn.LayerNorm(1024)
        self.norm_x_emb = nn.LayerNorm(1024) 
        
    def forward(self, x, x_emb):
        x = x.permute(0, 2, 1)
        # print(x.shape) #32, 1024, 2563
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape) #32, 256, 1281
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) #32, 128, 640
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape) #32, 64, 320
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x_emb = F.relu(self.fc1_emb(x_emb))
        x = self.norm_x(x)
        x_emb = self.norm_x_emb(x_emb)
        x = torch.cat((x, x_emb), dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def evaluate_model(model, test_loader, device):
    model.eval() 
    predictions, true_labels = [], []
    with torch.no_grad():
        for mut0, mut1, par0, labels in test_loader:
            mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)
            mut0_emb = moco_model(mut0)
            mut1_emb = moco_model(mut1)

            features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
            emb = torch.cat((mut0_emb, mut1_emb, mut0_emb - mut1_emb), dim=1).to(device)
            outputs = model(features, emb)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return true_labels, predictions

patience = 20
min_delta = 0.001
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
kf = KFold(n_splits=5, shuffle=True, random_state=0)

dataset = torch.load('data/embeddings/prottrans_embedding.pt')
for fold, (train_idx, test_val_idx) in enumerate(kf.split(dataset), 1):
    test_val_subset = Subset(dataset, test_val_idx)
    train_dataset = Subset(dataset, train_idx)

    test_size = int(len(test_val_subset) * 0.5)
    val_size = len(test_val_subset) - test_size
    test_dataset, val_dataset = random_split(test_val_subset, [test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
    model = SequenceModel(1024, 4).to(device)
    # model = TransformerClassifier(1, 1024, 1024, 2, 64, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss()
    moco_model = Embedding_CNN(1024).to(device)
    moco_model.load_state_dict(torch.load('model/contrastive_model_prottrans.pth'))
    moco_model.eval()

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0 

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters())

    # print(f"Total parameters moco: {count_parameters(moco_model)}")
    # print(f"Total parameters: {count_parameters(model)}")

    for epoch in tqdm(range(100), desc=f"Fold {fold}"):
        total_loss = 0
        model.train()
        for mut0, mut1, par0, labels in train_loader:
            mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)

            mut0_emb = moco_model(mut0)
            mut1_emb = moco_model(mut1)

            features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
            optimizer.zero_grad()

            outputs = model(features, torch.cat((mut0_emb, mut1_emb, mut0_emb - mut1_emb), dim=1).to(device))
            # print(outputs,labels)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Average Loss:', total_loss / len(train_loader))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mut0, mut1, par0, labels in val_loader:
                mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)
                mut0_emb = moco_model(mut0)
                mut1_emb = moco_model(mut1)

                features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
                outputs = model(features, torch.cat((mut0_emb, mut1_emb, mut0_emb - mut1_emb), dim=1).to(device))
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        print('val_accuracy:', val_accuracy)
        if val_accuracy - best_val_accuracy > min_delta:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch+1} due to no improvement.")
            break

    model.load_state_dict(best_model_state)

    true_labels, predictions = evaluate_model(model, test_loader, device)

    metrics['accuracy'].append(accuracy_score(true_labels, predictions))

    for metric_name, metric_func in zip(['precision', 'recall', 'f1'],[precision_score, recall_score, f1_score]):
        metric_values = metric_func(true_labels, predictions, average=None)
        metrics[metric_name].append(metric_values)

print("Average Test Performance across 5 folds:")
print(f'Accuracy: {np.mean(metrics["accuracy"])}')
for metric_name in ['precision', 'recall', 'f1']:
    print(f'{metric_name} for each class:')
    for i, value in enumerate(np.mean(metrics[metric_name], axis=0)):
        print(f'  Class {i}: {value}')
