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

class Obesity(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = './dataloader/datasets/obesity/'
        self.filename = 'Obesity.csv'

        # 加载和预处理数据
        df = pd.read_csv(os.path.join(self.root, self.filename))

        # 将二元分类特征转换为0和1
        binary_features = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        df[binary_features] = df[binary_features].apply(lambda x: x.map({'yes': 1, 'no': 0}))

        # 多类别特征进行标签编码
        multiclass_features = ['Gender', 'CAEC', 'CALC', 'MTRANS']
        label_encoders = {}
        for feature in multiclass_features:
            label_encoders[feature] = LabelEncoder()
            df[feature] = label_encoders[feature].fit_transform(df[feature])

        # 分离特征和标签
        self.y = df['NObeyesdad']
        df = df.drop('NObeyesdad', axis=1)

        # 对数值特征进行标准化
        numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        # 标签进行标签编码
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.y)

        # 加载数据
        self.X = torch.tensor(df.values, dtype=torch.float)
        self.target = torch.tensor(self.y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target

def prepare_dataset_obesity():
    # 创建Adult数据集实例
    dataset = Obesity()

    # 划分目标和影子数据集
    length = len(dataset)
    each_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, length - (each_length * 4)]
    )

    num_features = next(iter(dataset))[0].shape[0]
    print(num_features)
    num_classes = 7  # 输出类别数

    # 初始化模型
    target_model = Net_1(num_features, num_classes)
    shadow_model = Net_1(num_features, num_classes)

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model