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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# batch_size = 32
# dataset = torch.load('data/embeddings/prottrans_embedding.pt')

# train_size = int(0.7 * len(dataset))
# valid_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - valid_size

# train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, middle_dim):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, middle_dim)
        self.linear_2 = nn.Linear(middle_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, middle_dim):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, middle_dim)

    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        X2 = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens-1, 2, dtype=torch.float32) / (num_hiddens-1))
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X2)

    def forward(self, X):
        # print(X.shape, self.P[:, :X.shape[1], :].shape)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
# class PositionalEncoding(nn.Module):
#     def __init__(self, num_hiddens, dropout, max_len=1000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
        
#         # 初始化位置编码矩阵
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, num_hiddens, 2).float() * (-math.log(10000.0) / num_hiddens))
        
#         pe = torch.zeros(max_len, num_hiddens)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, X):
#         """
#         Args:
#             X: 输入的嵌入张量，形状为 [batch_size, seq_len, num_hiddens]
#         """
#         # 添加位置编码
#         print(X.shape, self.pe[:, :X.size(1)].shape)
#         X = X + self.pe[:, :X.size(1)].to(device)
#         return self.dropout(X)

class Embeddings(nn.Module):
    def __init__(self, ori_feature_dim, embed_dim):
        super().__init__()
        # self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.token_embeddings = nn.Linear(ori_feature_dim, embed_dim) # change embedding into linear for onehot
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.02)

    def forward(self, inputs, input_dim):
        #inputs = torch.tensor(inputs).to(inputs.device).long()
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(inputs)
        PE = PositionalEncoding(input_dim, 0)
        position_embeddings = PE(inputs)
        # Combine token and position embeddings
        # print(token_embeddings.shape, position_embeddings.shape)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerEncoder(nn.Module):
    def __init__(self, num_hidden_layers, vocab_size, embed_dim, num_heads, middle_dim, input_dim):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embed_dim)
        self.dim = input_dim
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, middle_dim)
                                     for _ in range(num_hidden_layers)])

        # self.dim_reducer1 = nn.Linear(embed_dim, 128)
        self.dim_reducer2 = nn.Linear(embed_dim, 32)


    def forward(self, x, mask=None):
        x = self.embeddings(x, self.dim)
        for layer in self.layers:
            x = layer(x, mask=mask)
        # print(x.shape)
        # x = self.dim_reducer1(x)
        x = self.dim_reducer2(x)  # 将维度从 [32, 513, 1024] 降至 [32, 513, 32]
        x = x.view(x.size(0), -1)  # 重塑为 [32, 16416]
        # print(x.shape)
        return x
    
# class TransformerClassifier(nn.Module):
#     def __init__(self, num_hidden_layers, vocab_size, embed_dim, num_heads, middle_dim, input_dim):
#         super().__init__()
#         self.embeddings = Embeddings(vocab_size, embed_dim)
#         self.dim = input_dim
#         self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, middle_dim)
#                                      for _ in range(num_hidden_layers)])

#         # self.dim_reducer1 = nn.Linear(embed_dim, 128)
#         self.dim_reducer2 = nn.Linear(embed_dim, 32)

#         self.fc1_emb = nn.Linear(513 * 32 * 3, 1024)
#         self.fc1 = nn.Linear(2563 * 32, 1024) 
#         self.fc2 = nn.Linear(2048, 256)
#         self.fc3 = nn.Linear(256, 4)
#         # self.norm_x = nn.LayerNorm(1024)  # 对x进行归一化
#         self.norm_x_emb = nn.LayerNorm(1024)  # 对x_emb进行归一化

#     def forward(self, x, x_emb, mask=None):
#         x = self.embeddings(x, self.dim)
#         for layer in self.layers:
#             x = layer(x, mask=mask)
#         # print(x.shape)
#         # x = self.dim_reducer1(x)
#         x = self.dim_reducer2(x)  # 将维度从 [32, 2563, 1024] 降至 [32, 2563, 32]
#         x = x.view(x.size(0), -1)  # 重塑为 [32, 82016]
#         # print(x.shape)
#         # print('x:',x)
#         # print('emb:', x_emb)

#         x = F.relu(self.fc1(x))
#         x_emb = F.relu(self.fc1_emb(x_emb))
#         # print('x1024:',x)
#         # print('emb1024:', x_emb)
#         # x = self.norm_x(x)
#         x_emb = self.norm_x_emb(x_emb)
#         # print('xnor:',x)
#         # print('embnor:', x_emb)
#         # print(x, x_emb)
#         x = torch.cat((x, x_emb), dim=1)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         # print(x.shape)
#         return x

class SequenceModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SequenceModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1_emb = nn.Linear(513 * 32 * 3, 1024)
        self.fc1 = nn.Linear(64 * 320, 1024)
        self.fc2 = nn.Linear(1024 * 2, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.norm_x = nn.LayerNorm(1024)  # 对x进行归一化
        self.norm_x_emb = nn.LayerNorm(1024)  # 对x_emb进行归一化
        
    def forward(self, x, x_emb):
        # print(x, x_emb)
        # 输入x的形状：[batch_size, seq_len, encoding_size]
        # 需要交换维度，因为Conv1d期望的输入形状是[batch_size, encoding_size, seq_len]
        x = x.permute(0, 2, 1)
        # print(x.shape) #32, 1024, 2563
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape) #32, 256, 1281
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) #32, 128, 640
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape) #32, 64, 320
        # 展平操作，为全连接层准备
        x = torch.flatten(x, 1)
        # print(x.shape)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x_emb = F.relu(self.fc1_emb(x_emb))

        x = self.norm_x(x)
        x_emb = self.norm_x_emb(x_emb)
        # print(x, x_emb)
        x = torch.cat((x, x_emb), dim=1)
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
    # 分割测试集和训练+验证集
    test_val_subset = Subset(dataset, test_val_idx)
    train_dataset = Subset(dataset, train_idx)
    
    # 验证集的50%数据作为测试集
    test_size = int(len(test_val_subset) * 0.5)
    val_size = len(test_val_subset) - test_size
    test_dataset, val_dataset = random_split(test_val_subset, [test_size, val_size])
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型和优化器
    model = SequenceModel(1024, 4).to(device)
    # model = TransformerClassifier(1, 1024, 1024, 2, 64, 1).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    moco_model = TransformerEncoder(1, 1024, 1024, 2, 64, 1).to(device)  # 创建模型实例
    moco_model.load_state_dict(torch.load('model/contrastive_model_1.pth'))  # 加载保存的state_dict
    moco_model.eval()  # 将模型设置为评估模式

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0 

    # 训练模型
    for epoch in tqdm(range(100), desc=f"Fold {fold}"):
        total_loss = 0
        model.train()
        for mut0, mut1, par0, labels in train_loader:
            mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)
            # print(mut0.device, next(moco_model.parameters()).device)
            mut0_emb = moco_model(mut0)
            mut1_emb = moco_model(mut1)
            # print(mut0, mut1, mut0-mut1)
            # print(mut0.shape, mut0_emb.shape)
            features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
            optimizer.zero_grad()
            # print(features.device, next(model.parameters()).device)
            outputs = model(features, torch.cat((mut0_emb, mut1_emb, mut0_emb - mut1_emb), dim=1).to(device))
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
                mut0_emb = moco_model(mut0)
                mut1_emb = moco_model(mut1)
                # print(mut0.shape, mut0_emb.shape)

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
    # for metric in metrics:
    #     print(f'{metric}: {np.mean(metrics[metric])}')
# 计算并打印平均性能指标
print("Average Test Performance across 5 folds:")
print(f'Accuracy: {np.mean(metrics["accuracy"])}')
for metric_name in ['precision', 'recall', 'f1']:
    print(f'{metric_name} for each class:')
    for i, value in enumerate(np.mean(metrics[metric_name], axis=0)):
        print(f'  Class {i}: {value}')
