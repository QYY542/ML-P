import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset

class KmeansDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_kmeans_distance(self, n_clusters=None):
        X_scaled = self.load_and_scale_data()

        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X_scaled)

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
    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()  # 假设X是numpy数组
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def find_optimal_clusters(self, X_scaled, max_clusters=10):
        sse = []
        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)
            sse.append(kmeans.inertia_)

        # Return the optimal number of clusters (elbow point)
        # This is a placeholder; in practice, you should visually identify the elbow point
        # or use an algorithmic approach
        optimal_clusters = sse.index(min(sse)) + 1
        return optimal_clusters
