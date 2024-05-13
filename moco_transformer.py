import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
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
import numpy as np

batch_size = 32
dataset = torch.load('data/embeddings/prottrans_embedding_013.pt')

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, num_hiddens, dropout, max_len=1000):
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
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features1, features2, negative_features):
        # 计算正样本对的相似度
        # print(features1.shape)
        positives = self.cosine_similarity(features1, features2).unsqueeze(-1)
        # 计算负样本对的相似度
        negatives = self.cosine_similarity(features1, negative_features).unsqueeze(-1)
        # print(positives.shape)
        # 将正负样本对的相似度拼接，并除以温度参数
        logits = torch.cat([positives, negatives], dim=-1) / self.temperature
        # 目标标签：正样本对的索引为0
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(features1.device)
        # print(logits.shape, labels.shape)
        # 使用交叉熵损失计算最终的对比损失
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
    
f_q = TransformerEncoder(1, 1024, 1024, 2, 64, 1).to(device)
optimizer = optim.Adam(f_q.parameters(), lr = 0.001)
contrastive_loss = ContrastiveLoss()

epoch_losses = []
num_epochs = 200
for epoch in range(num_epochs):
    total_loss = 0.0
    for mut0, mut1, par0, labels in loader:
        mut0, mut1 = mut0.to(device), mut1.to(device)
        optimizer.zero_grad()
        # print(batch[0].size(0))
        # if batch[0].size(0) < N:
        #     continue  # 跳过大小小于 batch_size 的批次
        x_q = mut0
        cropped_tensor = x_q[:, 64:-64, :]
        # print(cropped_tensor.shape, cropped_tensor[0])
        x_k = torch.nn.functional.pad(cropped_tensor, (0, 0, 64, 64), value=0.0).to(device)
        q = f_q(x_q)
        k = f_q(x_k)
        queue = f_q(mut1)
        # k = k.detach()
        # print(q.shape)
        loss = contrastive_loss(q, k, queue)
        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     for param_q, param_k in zip(f_q.parameters(), f_k.parameters()):
        #         param_k.data = param_k.data * m + param_q.data * (1-m)

            # batch_size = k.size(0)
            # print(queue[:, queue_ptr:queue_ptr + batch_size].shape, k.T.shape, batch_size, queue_ptr, queue_ptr + batch_size)
            # queue[:, queue_ptr:queue_ptr + batch_size] = k.T[:, :batch_size]
            # queue_ptr = (queue_ptr + batch_size) % K

        total_loss += loss.item() 

    average_loss = total_loss / len(loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')
    epoch_losses.append(average_loss)  # Store the average loss for this epoch

    if epoch == 0: 
        best_loss = average_loss
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(f_q.state_dict(), 'model/contrastive_model.pth')


# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
# plt.title('Epoch vs Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.savefig('training_loss_plot_013.png')  # Save the plot to a file
        
plt.figure(figsize=(10, 6))

sns.lineplot(x=range(1, num_epochs + 1), y=epoch_losses, marker="o", color="coral", lw=2, markersize=8, label='Loss')

plt.title('Contrastive Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.savefig('training_loss_plot_013_1.png')  # Save the plot to a file

















