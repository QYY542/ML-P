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



class Hospital(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = './dataloader/datasets/hospital/'
        self.filename = 'hospital.csv'

        # 加载和预处理数据
        df = pd.read_csv(os.path.join(self.root, self.filename))

        # 将目标标签转换为数值
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

        # 处理分类数据
        categorical_columns = ['race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
                               'admission_source_id', 'medical_specialty', 'payer_code', 'diag_1', 'diag_2', 'diag_3',
                               'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']
        df[categorical_columns] = df[categorical_columns].apply(
            lambda col: LabelEncoder().fit_transform(col.astype(str)))

        # 除了目标标签 'readmitted' 和已处理的分类列外，其余列都是数值型，可以直接标准化
        numeric_columns = df.columns.drop(categorical_columns + ['readmitted'])
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df[numeric_columns] = df[numeric_columns].fillna(0)  # 可以选择其他填充缺失值的方法

        # 对数值特征进行标准化
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        # 分离特征和标签
        X = df.drop('readmitted', axis=1).values
        target = df['readmitted'].values

        # 加载数据
        self.X = torch.tensor(X, dtype=torch.float)
        self.target = torch.tensor(target, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target

def prepare_dataset_hospital():
    # 创建Adult数据集实例
    dataset = Hospital()

    # 划分目标和影子数据集
    length = len(dataset)
    each_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, length - (each_length * 4)]
    )

    num_features = next(iter(dataset))[0].shape[0]
    print(num_features)
    num_classes = 3  # 输出类别数

    # 初始化模型
    target_model = Net_1(num_features,num_classes)
    shadow_model = Net_1(num_features,num_classes)

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model