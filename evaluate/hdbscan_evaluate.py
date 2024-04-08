import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
import hdbscan

class HDBSCANDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.min_cluster_size = self.calculate_min_cluster_size()

    def compute_hdbscan_clusters(self):
        X_scaled = self.load_and_scale_data()
        print("min_cluster_size = ", self.min_cluster_size)

        # 使用HDBSCAN进行聚类
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, gen_min_span_tree=True)
        clusterer.fit(X_scaled)
        labels = clusterer.labels_

        # 找到每个聚类的样本索引
        unique_labels = np.unique(labels[labels >= 0])  # 忽略噪声点
        cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

        return cluster_indices, X_scaled

    def apply_kmeans_to_clusters(self, cluster_indices, X_scaled):
        distances = np.zeros(len(X_scaled))

        for label, indices in cluster_indices.items():
            cluster_points = X_scaled[indices]
            kmeans = KMeans(n_clusters=1, random_state=42).fit(cluster_points)
            distances[indices] = np.linalg.norm(cluster_points - kmeans.cluster_centers_, axis=1)

        return distances

    def get_specific_datasets_and_distances(self, n):
        cluster_indices, X_scaled = self.compute_hdbscan_clusters()
        distances = self.apply_kmeans_to_clusters(cluster_indices, X_scaled)

        # 去掉0值样本
        distances_indices = np.where(distances > 0)[0]
        distances = distances[distances_indices]

        # 选择距离最远和最近的样本
        low_distance_indices = np.argsort(distances)[:n]
        high_distance_indices = np.argsort(distances)[-n:]
        random_indices = np.random.choice(len(self.dataset), n, replace=False)
        random_shadow_indices = np.random.choice(len(self.dataset), n + n, replace=False)

        excluded_indices = set(high_distance_indices).union(low_distance_indices, random_indices)
        remaining_indices = list(set(range(len(self.dataset))) - excluded_indices)

        test_indices = np.random.choice(remaining_indices, n, replace=False)

        # 获取对应的distance
        low_distance_values = distances[low_distance_indices]
        high_distance_values = distances[high_distance_indices]

        print("聚类距离小的样本分数:", low_distance_values)
        print("聚类距离大的样本分数:", high_distance_values)

        # 创建数据子集
        high_distance_dataset = Subset(self.dataset, high_distance_indices)
        low_distance_dataset = Subset(self.dataset, low_distance_indices)
        random_dataset = Subset(self.dataset, random_indices)
        test_dataset = Subset(self.dataset, test_indices)
        random_dataset_shadow = Subset(self.dataset, random_shadow_indices)

        return low_distance_dataset, high_distance_dataset, random_dataset, test_dataset, random_dataset_shadow

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def calculate_min_cluster_size(self):
        standard_size = 10000  # 标准大小
        base_min_cluster_size = 10  # 基础的min_cluster_size值

        # 获取数据集的大小和特征数
        data_size = len(self.dataset)
        # 假设dataset是二维的，例如：[样本数, 特征数]
        # 如果不是这样，需要根据实际情况调整
        num_features = next(iter(self.dataset))[0].shape[0]

        # 根据数据大小调整min_cluster_size
        size_factor = data_size / standard_size
        # 根据特征数量调整基础值
        feature_factor = 1 + (num_features - 1) / 20  # 假设每增加10个特征，min_cluster_size增加5%

        adjusted_min_cluster_size = base_min_cluster_size * max(size_factor, 1) * feature_factor
        return int(max(adjusted_min_cluster_size, 5))  # 确保min_cluster_size至少为5
