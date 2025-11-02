import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

class SequenceModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SequenceModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=5, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1_emb = nn.Linear(513 * 32 * 3, 1024)
        self.fc1 = nn.Linear(5 * 640, 1024)
        self.fc2 = nn.Linear(1024 , 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(x.shape) #32, 20, 2563
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape) #32, 10, 1281
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape) #32, 5, 640
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        # print(x.shape)
        return x
        
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for mut0, mut1, par0, labels in test_loader:
            mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)
            features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return true_labels, predictions

patience = 20
min_delta = 0.001
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
kf = KFold(n_splits=5, shuffle=True, random_state=0)

dataset = torch.load('data/embeddings/onehot_embedding.pt')
print(len(dataset))
for fold, (train_idx, test_val_idx) in enumerate(kf.split(dataset), 1):
    test_val_subset = Subset(dataset, test_val_idx)
    train_dataset = Subset(dataset, train_idx)
    
    test_size = int(len(test_val_subset) * 0.5)
    val_size = len(test_val_subset) - test_size
    test_dataset, val_dataset = random_split(test_val_subset, [test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = SequenceModel(20, 4).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0 

    for epoch in tqdm(range(100), desc=f"Fold {fold}"):
        total_loss = 0
        model.train()
        for mut0, mut1, par0, labels in train_loader:

            mut0, mut1, par0, labels = mut0.to(device), mut1.to(device), par0.to(device), labels.to(device)

            features = torch.cat((mut0, mut1, mut0-mut1, par0), dim=1).to(device)
            optimizer.zero_grad()
            outputs = model(features)
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
