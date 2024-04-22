import os
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler


class Hospital(Dataset):
    def __init__(self, filename, qid_indices=None) -> None:
        super().__init__()
        self.root = './dataloader/datasets/hospital/'
        self.filename = filename
        self.qid_indices = qid_indices

        # 加载数据
        df = pd.read_csv(os.path.join(self.root, self.filename))

        # 将目标标签转换为数值
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

        # 年龄处理，用区间中点代替区间值
        age_mapping = {f'[{10 * i}-{10 * (i + 1)})': 5 + 10 * i for i in range(10)}
        df['age'] = df['age'].map(age_mapping)

        # 随机填充缺失值
        missing_columns = ['race', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
        for column in missing_columns:
            if df[column].dtype == 'object':  # 对于分类数据
                existing_values = df[column].dropna().unique()
                df[column] = df[column].apply(lambda x: np.random.choice(existing_values) if pd.isnull(x) else x)
            else:  # 对于数值数据（如果有）
                range_min = df[column].min()
                range_max = df[column].max()
                df[column] = df[column].apply(lambda x: np.random.uniform(range_min, range_max) if pd.isnull(x) else x)

        # 对所有的分类数据使用LabelEncoder转换
        for column in df.select_dtypes(include=['object']).columns:
            lbl = LabelEncoder()
            df[column] = lbl.fit_transform(df[column])

        if qid_indices is not None:
            # 保留qid_indices指定的列以及Target列
            columns_to_keep = df.columns[qid_indices].tolist() + ['readmitted']
            df = df[columns_to_keep]

        # 分离特征和标签
        X = df.drop('readmitted', axis=1)
        target = df['readmitted'].values

        # 对数值特征进行标准化
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

        # 将处理后的特征转换为适用于PyTorch的格式
        self.X = torch.tensor(X.values, dtype=torch.float)
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
