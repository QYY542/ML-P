import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Callable, List, Optional, Union, Tuple

from models.define_models import  MLP


class Adult(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = './dataloader/datasets/adult/'
        self.filename = 'adult.csv'

        # 加载和预处理数据
        df = pd.read_csv(os.path.join(self.root, self.filename))
        df['income'] = df['income'].apply(lambda x: 0 if x == '<=50K' else 1)

        # 选择一些常用的属性作为例子
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                               'native-country']
        for column in categorical_columns:
            lbl = LabelEncoder()
            df[column] = lbl.fit_transform(df[column])

        # 分离特征和标签
        X = df.drop('income', axis=1).values
        target = df['income'].values

        # 对数值特征进行标准化
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 加载数据
        self.X = torch.tensor(X_scaled, dtype=torch.float)
        self.target = torch.tensor(target, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target


