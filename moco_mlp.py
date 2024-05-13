import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
import pandas as pd
import h5py
import math
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = h5py.File(file_path, "r")

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        data = torch.tensor(self.f[str(idx)][:])
        return data

# Create custom datasets
dataset1 = CustomDataset("/data/personal/liwh/macon/protT5/output/mut0_embeddings.h5")
dataset2 = CustomDataset("/data/personal/liwh/macon/protT5/output/mut1_embeddings.h5")

# Combine datasets into one dataset
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))

    def __getitem__(self, idx):
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]
        return data1, data2

# Create combined dataset
combined_dataset = CombinedDataset(dataset1, dataset2)

# Create DataLoader for combined dataset
batch_size = 32  # Set your desired batch size
loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 编码器 - 使用简单的池化层作为示例
class Encoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=128):
        super(Encoder, self).__init__()
        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，进一步压缩特征到更低维度
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x的形状：[batch_size, seq_length, feature_dim]
        # 调整x的形状以适应全局平均池化层
        x = x.permute(0, 2, 1).unsqueeze(-1)  # 形状变为[batch_size, feature_dim, seq_length, 1]
        # print(x.shape)
        x = self.global_avg_pool(x)  # 形状变为[batch_size, feature_dim, 1, 1]
        # print(x.shape)
        x = x.squeeze()  # 压缩最后两个维度，形状变为[batch_size, feature_dim]
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        return x
# 对比损失 - InfoNCE Loss
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

# 初始化模型和优化器
encoder = Encoder().to(device)
contrastive_loss = ContrastiveLoss()
optimizer = optim.Adam(encoder.parameters(), lr=0.001)

num_epochs = 100
# 示例训练循环
for epoch in range(num_epochs):
    for batch in loader:  # 假设 dataloader 是你的数据加载器
        optimizer.zero_grad()
        mut0_batch, mut1_batch = batch
        x_q = mut0_batch.to(device)
        cropped_tensor = mut0_batch[:, 64:-64, :]
        # print(cropped_tensor.shape, cropped_tensor[0])
        x_k = torch.nn.functional.pad(cropped_tensor, (0, 0, 64, 64), value=0.0).to(device)
        mut1_batch = mut1_batch.to(device)
        # print(x_q.shape)
        encoded_pos1 = encoder(x_q)
        encoded_pos2 = encoder(x_k)
        encoded_neg = encoder(mut1_batch)
        loss = contrastive_loss(encoded_pos1, encoded_pos2, encoded_neg)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
