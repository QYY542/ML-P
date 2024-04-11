from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Subset
import numpy as np
import hdbscan


# 添加噪音点的数据集训练出来的数据集隐私风险低
class HDBSCANDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        # self.min_cluster_size = self.calculate_min_cluster_size()
        self.min_cluster_size = 5

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
        print("min_cluster_size = ", self.min_cluster_size)

        # 使用HDBSCAN计算曼哈顿距离聚类
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, gen_min_span_tree=True, metric='manhattan')
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
                print(len(noise_indices))
                for index in noise_indices:
                    noise_point = X_scaled[index]
                    distances_to_centers = [np.sum(np.abs(noise_point - center)) for center in cluster_centers.values()]
                    # 对噪声点也应用距离调整因子
                    distances[index] = np.min(distances_to_centers) * (1 + distance_adjustment_factor[index])

        return distances

    def get_specific_datasets_and_distances(self, n):
        labels, X_scaled, probabilities = self.compute_hdbscan_clusters()
        distances = self.get_distances_and_probabilities(labels, X_scaled, probabilities)

        # 找出非噪声点的索引
        non_noise_indices = np.where(labels != -1)[0]
        # 找出噪声点的索引
        noise_indices = np.where(labels == -1)[0]

        # 只保留非噪声点的距离
        non_noise_distances = distances[non_noise_indices]
        # 对非噪声点的距离进行排序
        sorted_indices_tmp = np.argsort(non_noise_distances)
        # 根据排序后的索引获取实际的数据索引
        sorted_indices = non_noise_indices[sorted_indices_tmp]

        low_distance_indices = sorted_indices[:n]
        high_distance_indices = sorted_indices[-n:]

        # 对噪声点的距离进行处理
        if len(noise_indices) > 0:
            noise_distances = distances[noise_indices]
            noise_sorted_indices_tmp = np.argsort(-noise_distances)  # 对噪声点的距离进行降序排序
            noise_sorted_indices = noise_indices[noise_sorted_indices_tmp]
            # 选择距离最大的n个噪声点，如果噪声点不足n个，则选择所有噪声点
            selected_noise_indices = noise_sorted_indices[:min(len(noise_sorted_indices), n)]
            selected_noise_distances = noise_distances[noise_sorted_indices_tmp[:min(len(noise_sorted_indices), n)]]
        else:
            selected_noise_indices = []
            selected_noise_distances = []

        # 随机选择数据集和测试数据集
        random_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        test_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        random_shadow_indices = np.random.choice(range(len(self.dataset)), 2 * n, replace=False)

        # 创建对应的Subset
        low_distance_dataset = Subset(self.dataset, low_distance_indices)
        high_distance_dataset = Subset(self.dataset, high_distance_indices)
        noise_dataset = Subset(self.dataset, selected_noise_indices)  # 创建噪声点的数据集
        random_dataset = Subset(self.dataset, random_indices)
        test_dataset = Subset(self.dataset, test_indices)
        random_shadow_dataset = Subset(self.dataset, random_shadow_indices)

        low_distance_values = distances[low_distance_indices]
        high_distance_values = distances[high_distance_indices]
        random_shadow_values = distances[random_indices]

        print("簇内聚类距离小的样本分数:", low_distance_values)
        print("簇内聚类距离大的样本分数:", high_distance_values)
        print("噪音点距离最大的样本分数：", selected_noise_distances)
        print("聚类距离随机的样本分数:", random_shadow_values)

        return low_distance_dataset, high_distance_dataset, noise_dataset, random_dataset, test_dataset, random_shadow_dataset

    def calculate_min_cluster_size(self):
        standard_size = 10000  # 标准参考大小
        base_min_cluster_size = 5  # 调低基础的min_cluster_size值以提高灵活性
        data_size = len(self.dataset)  # 数据集的大小
        num_features = next(iter(self.dataset))[0].shape[0]  # 特征数量

        # 调整大小因子，确保对于较小的数据集，min_cluster_size不会过高
        size_factor = max(data_size / standard_size, 0.1)  # 确保即使是小数据集，size_factor也有一个最小值

        # 特征因子调整，对于每增加10个特征，min_cluster_size增加的比例稍微降低
        feature_factor = 1 + (num_features - 1) / 100  # 每增加100个特征，min_cluster_size仅增加1倍

        # 计算调整后的min_cluster_size
        adjusted_min_cluster_size = base_min_cluster_size * size_factor * feature_factor
        # 确保最终的min_cluster_size至少为5，同时也反映了基于数据大小和特征数量的动态调整
        return int(max(adjusted_min_cluster_size, 5))
