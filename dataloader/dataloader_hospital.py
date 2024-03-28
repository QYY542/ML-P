import os
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class Hospital(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = './dataloader/datasets/hospital/'
        self.filename = 'hospital.csv'

        # 加载数据
        df = pd.read_csv(os.path.join(self.root, self.filename))

        # 将目标标签转换为数值
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

        # 年龄处理，用区间中点代替区间值
        age_mapping = {f'[{10*i}-{10*(i+1)})': 5+10*i for i in range(10)}
        df['age'] = df['age'].map(age_mapping)

        # 分类数据处理
        categorical_columns = ['race', 'gender','weight', 'admission_type_id', 'discharge_disposition_id',
                               'admission_source_id', 'medical_specialty', 'payer_code', 'diag_1', 'diag_2', 'diag_3',
                               'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']

        # 对分类特征进行随机值填充
        for column in categorical_columns:
            if df[column].isnull().any():
                # 计算非缺失值的分布
                distribution = df[column].dropna().value_counts(normalize=True)
                # 生成随机抽样
                random_sampling = np.random.choice(distribution.index, size=df[column].isnull().sum(),
                                                   p=distribution.values)
                # 填充缺失值
                df[column][df[column].isnull()] = random_sampling

            lbl = LabelEncoder()
            df[column] = lbl.fit_transform(df[column])

        # 分离特征和标签
        X = df.drop('readmitted', axis=1)
        y = df['readmitted'].values

        # 对数值特征进行标准化
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

        # 将处理后的特征转换为张量
        self.X = torch.tensor(X.values, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target
