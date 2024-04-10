from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Subset
import numpy as np
import hdbscan


# 添加噪音点的数据集训练出来的数据集隐私风险低
class HDBSCANDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def compute_hdbscan_clusters(self):
        X_scaled = self.load_and_scale_data()
        X_scaled = X_scaled.astype(np.float64)

        # 使用HDBSCAN计算曼哈顿距离聚类
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='manhattan',
                                    algorithm='best', leaf_size=40)
        clusterer.fit(X_scaled)
        return clusterer.labels_, X_scaled, clusterer.probabilities_

    def get_distances_and_probabilities(self, labels, X_scaled, probabilities):
        unique_labels = np.unique(labels)
        cluster_centers = {label: X_scaled[labels == label].mean(axis=0) for label in unique_labels if label != -1}

        distances = np.zeros(len(X_scaled))  # 默认值设置为0
        # 为每个样本初始化一个距离调整因子，基于其归属概率
        distance_adjustment_factor = 1 - probabilities  # 归属概率越低，调整因子越大

        for label in unique_labels:
            if label != -1:
                cluster_points = X_scaled[labels == label]
                center = cluster_centers[label]
                # 计算曼哈顿距离，并应用距离调整因子
                adjusted_distances = np.sum(np.abs(cluster_points - center), axis=1) * (
                        1 + distance_adjustment_factor[labels == label])
                distances[labels == label] = adjusted_distances
            else:
                noise_indices = np.where(labels == -1)[0]
                for index in noise_indices:
                    noise_point = X_scaled[index]
                    distances_to_centers = [np.sum(np.abs(noise_point - center)) for center in cluster_centers.values()]
                    # 对噪声点也应用距离调整因子
                    distances[index] = np.min(distances_to_centers) * (1 + distance_adjustment_factor[index])

        return distances

    def get_specific_datasets_and_distances(self, n):
        labels, X_scaled, probabilities = self.compute_hdbscan_clusters()
        distances = self.get_distances_and_probabilities(labels, X_scaled, probabilities)

        # 排序并选择距离最小和最大的n个样本
        sorted_indices = np.argsort(distances)
        low_distance_indices = sorted_indices[:n]
        high_distance_indices = sorted_indices[-n:]

        # 随机选择数据集和测试数据集
        random_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        test_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        random_shadow_indices = np.random.choice(range(len(self.dataset)), 2 * n, replace=False)

        # 创建对应的Subset
        low_distance_dataset = Subset(self.dataset, low_distance_indices)
        high_distance_dataset = Subset(self.dataset, high_distance_indices)
        random_dataset = Subset(self.dataset, random_indices)
        test_dataset = Subset(self.dataset, test_indices)
        random_shadow_dataset = Subset(self.dataset, random_shadow_indices)

        low_distance_values = distances[low_distance_indices]
        high_distance_values = distances[high_distance_indices]
        random_shadow_values = distances[random_indices]

        print("聚类距离小的样本分数:", low_distance_values)
        print("聚类距离大的样本分数:", high_distance_values)
        print("聚类距离随机的样本分数:", random_shadow_values)

        return low_distance_dataset, high_distance_dataset, random_dataset, test_dataset, random_shadow_dataset

