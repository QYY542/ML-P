import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset


class KmeansDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_kmeans_distance(self, n_clusters=5):
        X_scaled = self.load_and_scale_data()

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

        # 创建排除了最小和最大聚类距离样本的随机数据集
        available_indices_for_min = list(set(range(len(self.dataset))) - set(min_indices))
        random_indices_shadow_min = np.random.choice(available_indices_for_min, n, replace=False)
        random_dataset_shadow_min = Subset(self.dataset, random_indices_shadow_min)

        available_indices_for_max = list(set(range(len(self.dataset))) - set(max_indices))
        random_indices_shadow_max = np.random.choice(available_indices_for_max, n, replace=False)
        random_dataset_shadow_max = Subset(self.dataset, random_indices_shadow_max)

        # 创建random_dataset的影子数据集，确保与random_dataset无重叠
        all_indices = set(range(len(self.dataset)))
        remaining_indices_for_shadow_random = list(all_indices - set(random_indices))
        random_indices_shadow_random = np.random.choice(remaining_indices_for_shadow_random, n, replace=False)
        random_dataset_shadow_random = Subset(self.dataset, random_indices_shadow_random)

        return min_dataset, max_dataset, random_dataset, random_dataset_shadow_min, random_dataset_shadow_max, random_dataset_shadow_random

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()  # 假设X是numpy数组
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
