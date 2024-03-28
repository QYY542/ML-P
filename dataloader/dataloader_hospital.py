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

        # 年龄处理，用区间中点代替区间值
        age_mapping = {f'[{10*i}-{10*(i+1)})': 5+10*i for i in range(10)}
        df['age'] = df['age'].map(age_mapping)

        # 分类数据处理
        categorical_columns = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id',
                               'admission_source_id', 'medical_specialty', 'payer_code', 'diag_1', 'diag_2', 'diag_3',
                               'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']

        # 使用众数填补分类特征的缺失值
        for column in categorical_columns:
            most_frequent = df[column].mode()[0]
            df[column] = df[column].fillna(most_frequent)
            lbl = LabelEncoder()
            df[column] = lbl.fit_transform(df[column])

        # 'weight'列特殊处理，此处选择填补为众数，考虑到您不想使用'Unknown'
        if 'weight' in df.columns:
            df['weight'] = df['weight'].fillna(df['weight'].mode()[0])
            lbl_weight = LabelEncoder()
            df['weight'] = lbl_weight.fit_transform(df['weight'])

        # 分离特征和标签
        X = df.drop('readmitted', axis=1).values
        target = df['readmitted'].values

        # 对数值特征进行标准化
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.difference(['readmitted'])
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        X_scaled = df[numeric_features].values

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
