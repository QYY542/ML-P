import numpy as np
from sklearn.metrics import silhouette_score
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
    def __init__(self, dataset):
        self.dataset = dataset

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def compute_hdbscan_clusters(self, min_cluster_size=5):
        X_scaled = self.load_and_scale_data()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', core_dist_n_jobs=-1)
        clusterer.fit(X_scaled)
        return clusterer.labels_, X_scaled, clusterer.probabilities_

    def get_distances_and_probabilities(self, labels, X_scaled, probabilities):
        unique_labels = np.unique(labels)
        # Calculate the distance to the nearest cluster center for each point
        distances = np.full(len(X_scaled), np.inf)
        for label in unique_labels:
            if label == -1: continue  # Skip noise for now
            cluster_points = X_scaled[labels == label]
            center = cluster_points.mean(axis=0)
            distances[labels == label] = np.linalg.norm(cluster_points - center, axis=1)
        # Consider noise points based on their probability (low probability = high risk)
        noise_indices = labels == -1
        distances[noise_indices] = np.max(distances) * (1 - probabilities[noise_indices]) # Scale distance for noise points by their outlier score
        return distances

    def get_specific_datasets_and_distances(self, n):
        labels, X_scaled, probabilities = self.compute_hdbscan_clusters()
        distances = self.get_distances_and_probabilities(labels, X_scaled, probabilities)

        # Use a more nuanced approach to determine high and low risk datasets
        sorted_indices = np.argsort(distances)  # From lowest to highest risk
        low_distance_indices = sorted_indices[:n]
        high_distance_indices = sorted_indices[-n:]
        random_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        test_indices = np.random.choice(range(len(self.dataset)), n, replace=False)
        random_shadow_indices = np.random.choice(range(len(self.dataset)), n + n, replace=False)

        # 获取对应的distance
        low_distance_values = distances[low_distance_indices]
        high_distance_values = distances[high_distance_indices]
        random_shadow_values = distances[random_indices]

        print("聚类距离小的样本分数:", low_distance_values)
        print("聚类距离大的样本分数:", high_distance_values)
        print("聚类距离随机的样本分数:", random_shadow_values)

        # Noise points considered as high risk, thus included in high_distance_dataset
        low_distance_indices = np.unique(np.concatenate([high_distance_indices, np.where(labels == -1)[0][:n]]))

        low_distance_dataset = Subset(self.dataset, low_distance_indices)
        high_distance_dataset = Subset(self.dataset, high_distance_indices)
        random_dataset = Subset(self.dataset, random_indices)
        test_dataset = Subset(self.dataset, test_indices)
        random_shadow_dataset = Subset(self.dataset, random_shadow_indices)

        return low_distance_dataset, high_distance_dataset, random_dataset, test_dataset, random_shadow_dataset

