import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Callable, List, Optional, Union, Tuple

from models.define_models import  Net_1


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

        X = df.drop('income', axis=1).values
        target = df['income'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 加载数据
        self.X = torch.tensor(X_scaled, dtype=torch.float)
        self.target = torch.tensor(target, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = self.X[index]
        target = self.target[index]

        return X, target


def prepare_dataset_adult():
    # 创建Adult数据集实例
    dataset = Adult()

    # 划分目标和影子数据集
    length = len(dataset)
    each_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, length - (each_length * 4)]
    )

    num_features = next(iter(dataset))[0].shape[0]
    print(num_features)
    num_classes = 2  # 输出类别数

    # 初始化模型
    target_model = Net_1(num_features,num_classes)
    shadow_model = Net_1(num_features,num_classes)

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model
