import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.nn.functional as F
import os
from torchvision import transforms
import pandas as pd
import h5py
import math
from tqdm import tqdm
import matplotlib.pyplot as plt 
from math import sqrt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# batch_size = 32

# train_size = int(0.7 * len(dataset))
# valid_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - valid_size

# train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SequenceModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SequenceModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=5, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1_emb = nn.Linear(513 * 32 * 3, 1024)
        self.fc1 = nn.Linear(5 * 640, 1024)  # 根据你的数据调整
        self.fc2 = nn.Linear(1024 , 256)
        self.fc3 = nn.Linear(256, num_classes)
        # self.norm_x = nn.LayerNorm(1024)  # 对x进行归一化
        # self.norm_x_emb = nn.LayerNorm(1024)  # 对x_emb进行归一化
        
    def forward(self, x):
        # print(x, x_emb)
        # 输入x的形状：[batch_size, seq_len, encoding_size]
        # 需要交换维度，因为Conv1d期望的输入形状是[batch_size, encoding_size, seq_len]
        x = x.permute(0, 2, 1)
        # print(x.shape) #32, 20, 2563
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape) #32, 10, 1281
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) #32, 5, 640
        # 展平操作，为全连接层准备
        x = torch.flatten(x, 1)
        # print(x.shape)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x_emb = F.relu(self.fc1_emb(x_emb))

        # x = self.norm_x(x)
        # x_emb = self.norm_x_emb(x_emb)
        # print(x, x_emb)
        # x = torch.cat((x, x_emb), dim=1)
        x = self.fc2(x)
        x = self.fc3(x)
        # print(x.shape)
        return x
        
def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions, true_labels = [], []
    with torch.no_grad():
        for mut0, mut1, par0, labels in test_loader:
            mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)
            # mut0_emb = moco_model(mut0)
            # mut1_emb = moco_model(mut1)

            features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
            # emb = torch.cat((mut0_emb, mut1_emb, mut0_emb - mut1_emb), dim=1).to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            # print(predicted,labels)
    return true_labels, predictions

patience = 20
min_delta = 0.001
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
kf = KFold(n_splits=5, shuffle=True, random_state=0)

dataset = torch.load('data/embeddings/onehot_embedding.pt')
print(len(dataset))
for fold, (train_idx, test_val_idx) in enumerate(kf.split(dataset), 1):
    # 分割测试集和训练+验证集
    test_val_subset = Subset(dataset, test_val_idx)
    train_dataset = Subset(dataset, train_idx)
    
    # 验证集的50%数据作为测试集
    test_size = int(len(test_val_subset) * 0.5)
    val_size = len(test_val_subset) - test_size
    test_dataset, val_dataset = random_split(test_val_subset, [test_size, val_size])
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 初始化模型和优化器
    model = SequenceModel(20, 4).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0 

    # 训练模型
    for epoch in tqdm(range(100), desc=f"Fold {fold}"):
        total_loss = 0
        model.train()
        for mut0, mut1, par0, labels in train_loader:

            mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)

            features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
            optimizer.zero_grad()
            # print(features.device, next(model.parameters()).device)
            outputs = model(features)
            # print(outputs,labels)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Average Loss:', total_loss / len(train_loader))
        # 验证模型
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mut0, mut1, par0, labels in val_loader:
                mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)
                # mut0_emb = moco_model(mut0)
                # mut1_emb = moco_model(mut1)
                # print(mut0.shape, mut0_emb.shape)

                features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        print('val_accuracy:', val_accuracy)
        if val_accuracy - best_val_accuracy > min_delta:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1  # 增加耐心计数器
        
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch+1} due to no improvement.")
            break

    # 加载最佳模型状态
    model.load_state_dict(best_model_state)

    # 在测试集上评估模型
    true_labels, predictions = evaluate_model(model, test_loader, device)

    metrics['accuracy'].append(accuracy_score(true_labels, predictions))

    for metric_name, metric_func in zip(['precision', 'recall', 'f1'],[precision_score, recall_score, f1_score]):
        metric_values = metric_func(true_labels, predictions, average=None)
        metrics[metric_name].append(metric_values)

        # metrics['accuracy'].append(accuracy_score(true_labels, predictions))
        # metrics['precision'].append(precision_score(true_labels, predictions, average='macro'))
        # metrics['recall'].append(recall_score(true_labels, predictions, average='macro'))
        # metrics['f1'].append(f1_score(true_labels, predictions, average='macro'))

# 计算并打印平均性能指标
print("Average Test Performance across 5 folds:")
print(f'Accuracy: {np.mean(metrics["accuracy"])}')
for metric_name in ['precision', 'recall', 'f1']:
    print(f'{metric_name} for each class:')
    for i, value in enumerate(np.mean(metrics[metric_name], axis=0)):
        print(f'  Class {i}: {value}')