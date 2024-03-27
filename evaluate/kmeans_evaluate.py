import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset

from dataloader.dataloader_obesity import Obesity
from main import test_meminf
from models.define_models import Net_1
from models.train_models import train_target_model, train_shadow_model


class KmeansDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_kmeans_distance(self, n_clusters=3):
        # 加载所有数据
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)

        for X, _ in loader:
            X = X.numpy()  # 假设X是numpy数组

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 训练k-means模型
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)

        # 计算到聚类中心的距离
        distances = kmeans.transform(X_scaled)
        min_distances = np.min(distances, axis=1)

        return min_distances

    def get_specific_datasets_and_distances(self, n):
        min_distances = self.compute_kmeans_distance()

        # 获取距离最小、最大和随机的索引
        min_indices = np.argsort(min_distances)[:n]
        max_indices = np.argsort(min_distances)[-n:]
        random_indices = np.random.choice(len(self.dataset), n, replace=False)

        # 获取对应的聚类距离
        min_distances_values = min_distances[min_indices]
        max_distances_values = min_distances[max_indices]
        random_distances_values = min_distances[random_indices]

        # 打印聚类距离
        print("聚类距离最小的样本距离:", min_distances_values)
        print("聚类距离最大的样本距离:", max_distances_values)
        print("随机选取的样本距离:", random_distances_values)

        # 根据索引创建数据子集
        min_dataset = Subset(self.dataset, min_indices)
        max_dataset = Subset(self.dataset, max_indices)
        random_dataset = Subset(self.dataset, random_indices)

        return min_dataset, max_dataset, random_dataset


def KmeansEvaluate(name):
    selected_dataset_name = name

    # 假设您已经正确加载了数据集
    dataset = Obesity("../dataloader/datasets/obesity/")  # 使用您的实际数据集
    num_classes = 7
    TARGET_PATH = "../dataloader/trained_model/" + "obesity" + selected_dataset_name + "Net_1"
    device = torch.device("cuda")

    # 取前三分之一样本的数据
    length = len(dataset)
    n = length // 3

    # 获取三类数据集 min max random
    evaluator = KmeansDataset(dataset)
    min_dataset, max_dataset, random_dataset = evaluator.get_specific_datasets_and_distances(n)

    if selected_dataset_name == "min":
        selected_dataset = min_dataset
    elif selected_dataset_name == "max":
        selected_dataset = max_dataset
    elif selected_dataset_name == "random":
        selected_dataset = random_dataset

    # 对min数据集进行分析
    selected_length = len(selected_dataset)
    each_selected_length = selected_length // 4
    num_features = next(iter(selected_dataset))[0].shape[0]
    min_target_train, min_target_test, min_shadow_train, min_shadow_test, _ = torch.utils.data.random_split(
        selected_dataset, [each_selected_length, each_selected_length, each_selected_length, each_selected_length,
                           selected_length - (each_selected_length * 4)]
    )

    # 获取模型并且评估
    target_model = Net_1(num_features, num_classes)
    shadow_model = Net_1(num_features, num_classes)

    train_target_model(TARGET_PATH, device, min_target_train, min_target_test, target_model)
    train_shadow_model(TARGET_PATH, device, min_shadow_train, min_shadow_test, shadow_model)
    test_meminf(TARGET_PATH, device, num_classes, min_target_train, min_target_test, min_shadow_train, min_shadow_test,
                target_model, shadow_model, mode=0, kmeans=selected_dataset_name)
