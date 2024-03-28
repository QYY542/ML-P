import os
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class Hospital_OneHot(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = './dataloader/datasets/hospital/'
        self.filename = 'hospital.csv'

        # 加载数据
        df = pd.read_csv(os.path.join(self.root, self.filename))

        # 将目标标签转换为数值
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

        # 处理年龄，使用区间的中点
        age_mapping = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
                       '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
        df['age'] = df['age'].map(age_mapping)

        # 定义预处理步骤
        categorical_columns = ['race', 'gender', 'weight', 'admission_type_id', 'discharge_disposition_id',
                               'admission_source_id', 'medical_specialty', 'payer_code', 'diag_1', 'diag_2', 'diag_3',
                               'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']

        # 创建预处理转换器
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
                (
                'num', StandardScaler(), ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                                          'number_outpatient', 'number_emergency', 'number_inpatient',
                                          'number_diagnoses'])
            ])

        # 分离特征和标签
        X = df.drop('readmitted', axis=1)
        y = df['readmitted']

        # 应用预处理
        X_processed = preprocessor.fit_transform(X)

        # 转换为PyTorch张量
        self.X = torch.tensor(X_processed.toarray(), dtype=torch.float)
        self.target = torch.tensor(y.values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        X = self.X[index]
        target: Any = []
        target.append(self.target[index])
        if target:
            target = tuple(target) if len(target) > 1 else target[0]

        return X, target
