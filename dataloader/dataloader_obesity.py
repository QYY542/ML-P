import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tarfile
import urllib
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
from models.define_models import MLP

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class Obesity(Dataset):
    def __init__(self, filename, qid_indices=None, DP=False) -> None:
        super().__init__()
        self.root = './dataloader/datasets/obesity/'
        self.filename = filename + '.csv'
        self.qid_indices = qid_indices
        self.epsilon = 1
        self.sensitivity = 0.5

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

        if qid_indices is not None:
            # 保留qid_indices指定的列以及Target列
            columns_to_keep = df.columns[qid_indices].tolist() + ['NObeyesdad']
            df = df[columns_to_keep]

        # 分离特征和标签
        self.y = df['NObeyesdad']
        df = df.drop('NObeyesdad', axis=1)

        # 对数值特征进行标准化
        numeric_features = df.columns  # 由于所有的特征都是数值型，我们可以直接使用所有列
        scaler = MinMaxScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        # 对qid_indices指定的敏感特征添加拉普拉斯噪声
        if DP:
            df = df.apply(lambda x: self.add_laplace_noise(x.values, self.epsilon, self.sensitivity))

        # 标签进行标签编码
        self.target_encoder = LabelEncoder()
        self.y = self.target_encoder.fit_transform(self.y)

        # 加载数据
        self.X = torch.tensor(df.values, dtype=torch.float)
        self.target = torch.tensor(self.y, dtype=torch.long)

    def add_laplace_noise(self, data, epsilon, sensitivity):
        # 添加拉普拉斯噪声以实现差分隐私。
        noise = np.random.laplace(loc=0.0, scale=sensitivity / epsilon, size=data.shape)
        return data + noise

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target
