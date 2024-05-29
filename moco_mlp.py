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
import math
from tqdm import tqdm
import matplotlib.pyplot as plt 
from math import sqrt
import seaborn as sns
import numpy as np
batch_size = 32
dataset = torch.load('data/embeddings/protbert_embedding.pt')
filtered_dataset = [(mut0, mut1, par0, labels) for mut0, mut1, par0, labels in dataset if labels != torch.tensor(2)]
print(len(filtered_dataset))
loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)
class SequenceModel(nn.Module):
    def __init__(self, input_size):
        super(SequenceModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1_emb = nn.Linear(513 * 32 * 3, 1024)
        self.fc1 = nn.Linear(64 * 64, 1024)  # 根据你的数据调整
        # self.norm_x = nn.LayerNorm(1024)  # 对x进行归一化
        # self.norm_x_emb = nn.LayerNorm(1024)  # 对x_emb进行归一化
        
    def forward(self, x):
        # print(x, x_emb)
        # 输入x的形状：[batch_size, seq_len, encoding_size]
        # 需要交换维度，因为Conv1d期望的输入形状是[batch_size, encoding_size, seq_len]
        x = x.permute(0, 2, 1)
        # print(x.shape) #32, 1024, 513
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape) #32, 256, 256
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) #32, 128, 128
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape) #32, 64, 64
        # 展平操作，为全连接层准备
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
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
    
f_q = SequenceModel(1024).to(device)
optimizer = optim.Adam(f_q.parameters(), lr = 0.001)
contrastive_loss = ContrastiveLoss()

epoch_losses = []
num_epochs = 50
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
        torch.save(f_q.state_dict(), 'model/contrastive_model_protbert.pth')


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
plt.savefig('training_loss_plot_protbert.png')  # Save the plot to a file

















