import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
batch_size = 32
dataset = torch.load('data/embeddings/protbert_embedding_pretrain.pt')
# print(len(filtered_dataset))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)
class SequenceModel(nn.Module):
    def __init__(self, input_size):
        super(SequenceModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64, 1024)
        
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
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features1, features2, negative_features):
        positives = self.cosine_similarity(features1, features2).unsqueeze(-1)
        negatives = self.cosine_similarity(features1, negative_features).unsqueeze(-1)
        logits = torch.cat([positives, negatives], dim=-1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(features1.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
    
f_q = SequenceModel(1024).to(device)
optimizer = optim.Adam(f_q.parameters(), lr = 0.001)
contrastive_loss = ContrastiveLoss()

epoch_losses = []
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0.0
    for mut0, mut1, par0, labels in loader:
        mut0, mut1 = mut0.to(device), mut1.to(device)
        optimizer.zero_grad()
        x_q = mut0
        cropped_tensor = x_q[:, 64:-64, :]
        x_k = torch.nn.functional.pad(cropped_tensor, (0, 0, 64, 64), value=0.0).to(device)
        q = f_q(x_q)
        k = f_q(x_k)
        queue = f_q(mut1)
        loss = contrastive_loss(q, k, queue)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() 

    average_loss = total_loss / len(loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')
    epoch_losses.append(average_loss)  

    if epoch == 0: 
        best_loss = average_loss
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(f_q.state_dict(), 'model/contrastive_model_prothert.pth')
        
plt.figure(figsize=(10, 6))

sns.lineplot(x=range(1, num_epochs + 1), y=epoch_losses, marker="o", color="coral", lw=2, markersize=8, label='Loss')

plt.title('Contrastive Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.savefig('training_loss_plot_protbert.png') 

