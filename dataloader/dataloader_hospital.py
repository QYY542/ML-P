import os
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
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

        # 处理缺失值
        df.replace('?', 'Unknown', inplace=True)

        # 特殊处理列'weight'，因为它大部分是缺失的
        df['weight'] = 'Unknown'

        # 年龄处理，用区间中点代替区间值
        age_mapping = {f'[{10 * i}-{10 * (i + 1)})': 5 + 10 * i for i in range(10)}
        df['age'] = df['age'].map(age_mapping)

        # 处理分类数据
        categorical_columns = ['race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
                               'admission_source_id', 'medical_specialty', 'payer_code', 'diag_1', 'diag_2', 'diag_3',
                               'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']
        for column in categorical_columns:
            df[column] = df[column].fillna('Unknown')  # 先填充缺失值为'Unknown'
            lbl = LabelEncoder()
            df[column] = lbl.fit_transform(df[column])

        # 分离特征和标签
        X = df.drop('readmitted', axis=1).values
        target = df['readmitted'].values

        # 对数值特征进行标准化
        scaler = StandardScaler()
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
