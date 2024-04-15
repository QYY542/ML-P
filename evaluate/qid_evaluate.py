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

    def train_model(self, X, y, n_estimators=50):
        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, oob_score=True)
        model.fit(X, y)
        return model

    def compute_qid_impacts(self, qid_index, num_trials=30):
        # 获取训练数据
        train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        X_train, y_train = next(iter(train_loader))
        X_train = X_train.numpy()
        y_train = y_train.numpy()

        # 训练模型并开启OOB评分
        model = self.train_model(X_train, y_train)

        # 使用OOB样本进行Permutation Importance计算
        oob_permutation_scores = []
        for _ in range(num_trials):
            X_train_shuffled = X_train.copy()
            np.random.shuffle(X_train_shuffled[:, qid_index])  # 打乱特定特征

            # 检查哪些样本是OOB样本（oob_decision_function_ 中有NaN的是OOB样本）
            oob_mask = np.isnan(model.oob_decision_function_[:, 1]) == False

            # 原始OOB得分
            oob_score_original = accuracy_score(y_train[oob_mask], model.predict(X_train)[oob_mask])
            oob_score_shuffled = accuracy_score(y_train[oob_mask], model.predict(X_train_shuffled)[oob_mask])
            impact = oob_score_original - oob_score_shuffled
            oob_permutation_scores.append(impact)

        # 返回平均影响值
        return np.mean(oob_permutation_scores)

