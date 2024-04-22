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
        # 划分数据集为训练集和测试集
        indices = np.arange(len(self.dataset))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

    def train_model(self, X, y, n_estimators=100):
        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X, y)
        return model

    def get_test_data(self):
        # 获取测试集数据
        test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
        X_test, y_test = next(iter(test_loader))
        return X_test.numpy(), y_test.numpy()

    def compute_qid_impacts(self, qid_index, num_trials=30):
        # 获取训练和测试数据
        train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        X_train, y_train = next(iter(train_loader))
        X_train = X_train.numpy()
        y_train = y_train.numpy()
        X_test, y_test = self.get_test_data()

        # 训练模型
        model = self.train_model(X_train, y_train)

        # 使用测试集进行Permutation Importance计算
        permutation_scores = []
        for _ in range(num_trials):
            X_test_shuffled = X_test.copy()
            np.random.shuffle(X_test_shuffled[:, qid_index])  # 打乱特定特征

            score_original = accuracy_score(y_test, model.predict(X_test))
            score_shuffled = accuracy_score(y_test, model.predict(X_test_shuffled))
            impact = score_original - score_shuffled
            permutation_scores.append(impact)

        # 返回平均影响值
        return np.mean(permutation_scores)

    def evaluate_model(self):
        # 获取测试数据
        X_test, y_test = self.get_test_data()

        # 获取训练好的模型
        train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        X_train, y_train = next(iter(train_loader))
        X_train = X_train.numpy()
        y_train = y_train.numpy()
        model = self.train_model(X_train, y_train)

        # 计算准确率
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

