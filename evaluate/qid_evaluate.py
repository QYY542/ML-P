import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataloader.dataloader_obesity import Obesity
from dataloader.dataloader_student import Student

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from torch.utils.data import DataLoader, Subset


class QID_VE:
    def __init__(self, dataset):
        self.test_dataset = None
        self.train_dataset = None
        self.dataset = dataset

    def train_test_split(self, test_size=0.3):
        indices = np.arange(len(self.dataset))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

    def train_model(self, X, y, n_estimators=50):
        # 根据给定的数据训练模型
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
        return model

    def compute_qid_impacts(self, qid_index, num_trials=10):
        # 获取训练数据
        train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        X_train, y_train = next(iter(train_loader))
        X_train = X_train.numpy()
        y_train = y_train.numpy()

        # 训练原始模型并计算评分
        model_original = self.train_model(X_train, y_train)
        score_original = cross_val_score(model_original, X_train, y_train, cv=5).mean()

        impacts = []
        for i in range(num_trials):
            # 打乱特定特征
            X_train_shuffled = X_train.copy()

            if i > num_trials // 2:
                # 生成与特征长度相同的0-1之间的随机值来填充
                random_values = np.random.uniform(0, 1, X_train_shuffled[:, qid_index].shape)
                X_train_shuffled[:, qid_index] = random_values
            elif i < num_trials // 2:
                # 打乱特定特征
                original_values = X_train[:, qid_index]
                shuffled_values = np.random.choice(original_values, size=original_values.shape, replace=True)
                X_train_shuffled[:, qid_index] = shuffled_values

            # 训练打乱特征后的模型并计算评分
            model_shuffled = self.train_model(X_train_shuffled, y_train)
            score_shuffled = cross_val_score(model_shuffled, X_train_shuffled, y_train, cv=5).mean()

            # 计算影响力
            impact = score_original - score_shuffled
            impacts.append(impact)

        # 返回多次试验的平均impact值
        return np.mean(impacts)

