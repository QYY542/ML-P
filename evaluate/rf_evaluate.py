import torch
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Any

from dataloader.dataloader_obesity import Obesity


# 假设Student类和Obesity类已经被正确定义

class QID_VE:
    def __init__(self, dataset):
        self.dataset = dataset

    def train_test_split(self, test_size=0.3):
        indices = np.arange(len(self.dataset))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)



if __name__ == '__main__':
    # 加载数据集
    # dataset = Student("../dataloader/datasets/student/")
    dataset = Obesity("../dataloader/datasets/obesity/")
    num_features = next(iter(dataset))[0].shape[0]

    evaluator = QID_VE(dataset)
    evaluator.train_test_split()

