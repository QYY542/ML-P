import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
import hdbscan


class HDBSCANDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_hdbscan_clusters(self):
        X_scaled = self.load_and_scale_data()

        # 使用HDBSCAN进行聚类
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        clusterer.fit(X_scaled)

        # HDBSCAN不直接提供到聚类中心的距离，但我们可以使用membership_vector_属性
        # 来获取每个点属于其聚类的程度（这个值越高，点越可能属于聚类）
        # 注意：这个属性可能不总是可用，取决于HDBSCAN的参数设置
        scores = clusterer.probabilities_

        # 将这些“距离”值用于后续的逻辑
        return scores

    def get_specific_datasets_and_distances(self, n):
        scores = self.compute_hdbscan_clusters()

        # 由于我们没有直接的距离度量，我们使用scores表示点属于聚类的程度
        # 最高分和最低分（如果有意义的话），或者选择其他逻辑
        low_scores_indices = np.argsort(scores)[:n]
        high_scores_indices = np.argsort(scores)[-n:]
        random_indices = np.random.choice(len(self.dataset), n, replace=False)
        random_shadow_indices = np.random.choice(len(self.dataset), n + n, replace=False)

        # 获取对应的scores
        high_scores_values = scores[high_scores_indices]
        low_scores_values = scores[low_scores_indices]

        print("属于聚类程度最低的样本分数:", low_scores_values)
        print("属于聚类程度最高的样本分数:", high_scores_values)


        # 根据索引创建数据子集
        high_score_dataset = Subset(self.dataset, high_scores_indices)
        low_score_dataset = Subset(self.dataset, low_scores_indices)
        random_dataset = Subset(self.dataset, random_indices)
        random_dataset_shadow = Subset(self.dataset, random_shadow_indices)

        # 排除min_dataset, max_dataset, random_dataset的索引
        excluded_indices = set(high_scores_indices).union(low_scores_indices, random_indices)
        all_indices = set(range(len(self.dataset)))
        remaining_indices = list(all_indices - excluded_indices)

        # 从剩余索引中选择随机数据集作为test_dataset
        test_indices = np.random.choice(remaining_indices, n, replace=False)
        test_dataset = Subset(self.dataset, test_indices)

        return low_score_dataset, high_score_dataset, random_dataset, test_dataset, random_dataset_shadow

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()  # 假设X是numpy数组
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
