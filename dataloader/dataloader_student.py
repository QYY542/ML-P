import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tarfile
import urllib
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
from models.define_models import  Net_1

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler



class Student(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = './dataloader/datasets/student/'
        self.filename = 'student.csv'

        # 加载和预处理数据
        df = pd.read_csv(os.path.join(self.root, self.filename))

        # 分离特征和标签
        self.y = df['Target']
        df = df.drop('Target', axis=1)

        # 数值特征进行标准化
        numeric_features = df.columns  # 由于所有的特征都是数值型，我们可以直接使用所有列
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        # 标签进行编码
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.y)

        # 加载数据
        self.X = torch.tensor(df.values, dtype=torch.float)
        self.target = torch.tensor(self.y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target

def prepare_dataset_student():
    # 创建Adult数据集实例
    dataset = Student()

    # 划分目标和影子数据集
    length = len(dataset)
    each_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, length - (each_length * 4)]
    )

    num_features = next(iter(dataset))[0].shape[0]
    print(each_length)
    print(num_features)
    num_classes = 3  # 输出类别数

    # 初始化模型
    target_model = Net_1(num_features, num_classes)
    shadow_model = Net_1(num_features, num_classes)

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model