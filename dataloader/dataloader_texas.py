import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tarfile
import urllib
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
from models.define_models import TexasClassifier


class Texas(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.root = './dataloader/datasets/texas/'
        self.features_path = os.path.join(self.root, 'feats.csv')
        self.labels_path = os.path.join(self.root, 'labels.csv')

        df_features = pd.read_csv(self.features_path)
        df_labels = pd.read_csv(self.labels_path)

        np_features = df_features.values
        np_labels = df_labels.values.squeeze()

        # 加载数据
        self.X = torch.tensor(np_features, dtype=torch.float)
        self.target = torch.tensor(np_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = self.X[index]
        target = self.target[index]

        return X, target


def prepare_dataset_texas():
    # 创建Adult数据集实例
    dataset = Texas()

    # 划分目标和影子数据集
    length = len(dataset)
    each_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, length - (each_length * 4)]
    )

    num_features = next(iter(dataset))[0].shape[0]
    print(num_features)
    num_classes = 100  # 输出类别数

    # 初始化模型
    target_model = TexasClassifier(num_features)
    shadow_model = TexasClassifier(num_features)

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model
