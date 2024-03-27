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

    def train_model(self, n_estimators=400):
        # 返回训练好的模型
        train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        X_train, y_train = next(iter(train_loader))
        X_train = X_train.numpy()
        y_train = y_train.numpy()

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        return model

    def compute_qid_impacts(self, qid_index, num_trials=20):
        impacts = []
        X_train, y_train = next(iter(DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)))
        X_train = X_train.numpy()
        y_train = y_train.numpy()

        for _ in range(num_trials):
            X_train_shuffled = X_train.copy()
            np.random.shuffle(X_train_shuffled[:, qid_index])

            model_original = self.train_model()
            model_shuffled = self.train_model()

            score_original = cross_val_score(model_original, X_train, y_train, cv=5).mean()
            score_shuffled = cross_val_score(model_shuffled, X_train_shuffled, y_train, cv=5).mean()

            impact = score_original - score_shuffled
            impacts.append(impact)

        return np.mean(impacts)  # 返回多次试验的平均impact值


if __name__ == '__main__':
    # 假设Student或Obesity类已经定义
    dataset = Student("../dataloader/datasets/student/")
    # dataset = Obesity("../dataloader/datasets/obesity/")

    evaluator = QID_VE(dataset)
    evaluator.train_test_split()

    num_features = next(iter(dataset))[0].shape[0]

    for i in range(num_features):
        impact = evaluator.compute_qid_impacts(i)
        print(f"Impact for QID {i}: {impact}")
