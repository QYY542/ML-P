import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataloader.dataloader_obesity import Obesity
from dataloader.dataloader_student import Student


class QID_VE:
    def __init__(self, dataset):
        self.dataset = dataset

    def train_test_split(self, test_size=0.3):
        indices = np.arange(len(self.dataset))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

    def train_model(self, n_estimators=400):
        train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        X_train, y_train = next(iter(train_loader))
        X_train = X_train.numpy()
        y_train = y_train.numpy()

        self.model = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=42)
        self.model.fit(X_train, y_train)

    def compute_qid_impacts(self, qid_index, num_trials=20):
        X_train, y_train = next(iter(DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)))
        X_train = X_train.numpy()
        y_train = y_train.numpy()

        impacts = []

        for _ in range(num_trials):
            X_train_shuffled = X_train.copy()
            np.random.shuffle(X_train_shuffled[:, qid_index])

            model_shuffled = RandomForestClassifier(n_estimators=self.model.n_estimators, oob_score=True,
                                                    random_state=42)
            model_shuffled.fit(X_train_shuffled, y_train)

            impact = self.model.oob_score_ - model_shuffled.oob_score_
            impacts.append(impact)

        return impacts

    def compute_qid_impact_std_deviation(self, qid_index, num_trials=5):
        impacts = self.compute_qid_impacts(qid_index, num_trials)
        average_impact = np.mean(impacts)
        std_deviation = np.std(impacts)

        return average_impact, std_deviation


if __name__ == '__main__':
    # 假设Student或Obesity类已经定义
    dataset = Student("../dataloader/datasets/student/")
    # dataset = Obesity("../dataloader/datasets/obesity/")

    evaluator = QID_VE(dataset)
    evaluator.train_test_split()
    evaluator.train_model()  # 训练模型

    num_features = next(iter(dataset))[0].shape[0]
    print(next(iter(dataset))[0])

    for i in range(num_features):
        impact, std_deviation = evaluator.compute_qid_impact_std_deviation(i)
        print(f"Standard deviation of impact for QID {i}: {std_deviation}")
        print(f"{impact}")
        print(f"{std_deviation/impact}")
