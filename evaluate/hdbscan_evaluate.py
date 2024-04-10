import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
import hdbscan

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, Subset
import numpy as np
import hdbscan

class HDBSCANDataset:
    def __init__(self, dataset, noise_handling='none'):  # 添加一个参数来控制噪声点的处理方式
        self.dataset = dataset
        self.noise_handling = noise_handling

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def compute_hdbscan_clusters(self, min_cluster_size=30):
        X_scaled = self.load_and_scale_data()

        # 确保X_scaled是float64类型
        X_scaled = X_scaled.astype(np.float64)

        # 如果使用余弦距离作为度量，则需要将X_scaled转换为距离矩阵
        # 注意：转换成距离矩阵后，数据类型应该保持为float64
        distance_matrix = cosine_distances(X_scaled).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    metric='precomputed',  # 如果你使用的是预计算的距离矩阵
                                    core_dist_n_jobs=-1)
        clusterer.fit(distance_matrix)
        return clusterer.labels_, X_scaled, clusterer.probabilities_

    def get_distances_and_probabilities(self, labels, X_scaled, probabilities):
        unique_labels = np.unique(labels)
        distances = np.full(len(X_scaled), np.inf)
        for label in unique_labels:
            if label == -1:
                continue  # 现在我们跳过噪声点的处理，稍后单独处理
            cluster_points = X_scaled[labels == label]
            center = cluster_points.mean(axis=0)
            distances[labels == label] = np.linalg.norm(cluster_points - center, axis=1)

        # 将噪声点的距离设置为-1
        noise_indices = labels == -1
        distances[noise_indices] = -1
        return distances

    def get_specific_datasets_and_distances(self, n):
        labels, X_scaled, probabilities = self.compute_hdbscan_clusters()
        distances = self.get_distances_and_probabilities(labels, X_scaled, probabilities)

        # 仅选择距离大于0的索引作为候选
        candidate_indices = distances > 0
        sorted_indices = np.argsort(distances[candidate_indices])
        actual_indices = np.arange(len(distances))[candidate_indices][sorted_indices]  # 转换回原始索引

        low_distance_indices = actual_indices[:n]
        high_distance_indices = actual_indices[-n:]

        # 根据noise_handling参数决定是否添加噪声点
        noise_indices = np.where(labels == -1)[0]
        if self.noise_handling == 'low':
            low_distance_indices = np.unique(np.concatenate([low_distance_indices, noise_indices]))
        elif self.noise_handling == 'high':
            high_distance_indices = np.unique(np.concatenate([high_distance_indices, noise_indices]))

        # 随机数据集和测试数据集的生成不需要改变
        random_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        test_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        random_shadow_indices = np.random.choice(range(len(self.dataset)), n + n, replace=False)

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