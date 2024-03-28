import os
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
import tarfile
import urllib
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
from models.define_models import Net_1

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class Hospital(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = './dataloader/datasets/hospital/'
        self.filename = 'hospital.csv'

        # 加载数据
        df = pd.read_csv(os.path.join(self.root, self.filename))

        # 将'?'替换为NaN以便于处理
        df.replace('?', np.nan, inplace=True)

        # 将目标标签转换为数值
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

        # 定义数值和分类特征
        numeric_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                           'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
        categorical_columns = ['race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
                               'admission_source_id', 'medical_specialty', 'payer_code', 'diag_1', 'diag_2', 'diag_3',
                               'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']

        # 处理缺失值
        df[numeric_columns] = SimpleImputer(strategy='median').fit_transform(df[numeric_columns])
        df[categorical_columns] = SimpleImputer(strategy='constant', fill_value='missing').fit_transform(
            df[categorical_columns])

        # 标签编码分类特征
        for col in categorical_columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # 标准化数值特征
        df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

        # 分离特征和标签
        X = df.drop('readmitted', axis=1).values
        y = df['readmitted'].values

        # 转换为torch.tensor
        self.X = torch.tensor(X, dtype=torch.float)
        self.target = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target
