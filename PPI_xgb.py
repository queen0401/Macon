import torch
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载数据
data = torch.load('data/embeddings/prottrans_embedding.pt')

# 数据预处理：转换 Tensor 为 NumPy 数组
features = []
labels = []
for item in data:
    mut0, mut1, par0, label = item['mut0'], item['mut1'], item['par0'], item['labels']
    feature = torch.cat((mut0, mut1, mut0 - mut1, par0), dim=0).numpy()
    features.append(feature)
    labels.append(label.numpy())

features = np.array(features)
labels = np.array(labels).reshape(-1)

# 设置交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for fold, (train_idx, test_val_idx) in enumerate(kf.split(features), 1):
    # 数据集划分
    X_train, X_test_val = features[train_idx], features[test_val_idx]
    y_train, y_test_val = labels[train_idx], labels[test_val_idx]

    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
    
    # 初始化 XGBoost 模型
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.05, random_state=42)
    
    # 训练模型
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average=None))
    metrics['recall'].append(recall_score(y_test, y_pred, average=None))
    metrics['f1'].append(f1_score(y_test, y_pred, average=None))

# 计算并打印平均性能指标
print("Average Test Performance across 5 folds:")
print(f'Accuracy: {np.mean(metrics["accuracy"])}')
for metric_name in ['precision', 'recall', 'f1']:
    print(f'{metric_name} for each class:')
    for i, value in enumerate(np.mean(metrics[metric_name], axis=0)):
        print(f'  Class {i}: {value}')
