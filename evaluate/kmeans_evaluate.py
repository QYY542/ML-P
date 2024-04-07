import pickle
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset

from evaluate.mia_evaluate import attack_for_blackbox
from models.define_models import ShadowAttackModel


class KmeansDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_kmeans_distance(self):
        n_clusters = self.elbow_method()  # 使用肘部方法确定最佳聚类数
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
        random_shadow_indices = np.random.choice(len(self.dataset), n, replace=False)

        # 获取对应的聚类距离
        min_distances_values = min_distances[min_indices]
        max_distances_values = min_distances[max_indices]
        random_distances_values = min_distances[random_indices]
        random_shadow_distances_values = min_distances[random_shadow_indices]

        # 打印聚类距离
        print("聚类距离最小的样本距离:", min_distances_values)
        print("聚类距离最大的样本距离:", max_distances_values)
        print("随机选取的样本距离:", random_distances_values)
        print("随机选取影子的样本距离:", random_shadow_distances_values)

        # 根据索引创建数据子集
        min_dataset = Subset(self.dataset, min_indices)
        max_dataset = Subset(self.dataset, max_indices)
        random_dataset = Subset(self.dataset, random_indices)
        random_dataset_shadow = Subset(self.dataset, random_shadow_indices)

        return min_dataset, max_dataset, random_dataset, random_dataset_shadow

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()  # 假设X是numpy数组
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def elbow_method(self, max_clusters=50):
        X_scaled = self.load_and_scale_data()
        sse = []
        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)
            sse.append(kmeans.inertia_)  # inertia_是模型的SSE

        # 寻找肘部点
        k_optimal = self.find_elbow_point(sse)
        return k_optimal

    def find_elbow_point(self, sse):
        # 这里简化处理，返回SSE下降变缓的第一个点作为肘部
        # 更复杂的实现可能需要计算曲率等
        for i in range(1, len(sse) - 1):
            if (sse[i] - sse[i + 1]) < (sse[i - 1] - sse[i]) / 2:
                return i + 1
        return len(sse)  # 如果没有找到明显的肘点，返回最大聚类数



def evaluate_attack_model(model_path, test_set_path, result_path, num_classes, epoch):
    # 加载攻击模型
    attack_model = ShadowAttackModel(num_classes)
    attack_model.load_state_dict(torch.load(model_path, map_location=attack_model.device))
    attack_model.eval()

    correct = 0
    total = 0
    final_test_ground_truth = []
    final_test_prediction = []
    final_test_probability = []

    with torch.no_grad():
        with open(test_set_path, "rb") as f:
            while True:
                try:
                    output, prediction, members = pickle.load(f)
                    # 确保数据在正确的设备上
                    output, prediction, members = output.to(attack_model.device), prediction.to(
                        attack_model.device), members.to(attack_model.device)

                    results = attack_model(output, prediction)
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()
                    probabilities = F.softmax(results, dim=1)

                    if epoch:  # 如果有epoch参数，保存详细结果
                        final_test_ground_truth.append(members)
                        final_test_prediction.append(predicted)
                        final_test_probability.append(probabilities[:, 1])

                except EOFError:
                    break

    if epoch:  # 处理和保存测试结果
        final_test_ground_truth = torch.cat(final_test_ground_truth, dim=0).cpu().numpy()
        final_test_prediction = torch.cat(final_test_prediction, dim=0).cpu().numpy()
        final_test_probability = torch.cat(final_test_probability, dim=0).cpu().numpy()

        test_f1_score = f1_score(final_test_ground_truth, final_test_prediction)
        test_roc_auc_score = roc_auc_score(final_test_ground_truth, final_test_probability)

        with open(result_path, "wb") as f:
            pickle.dump((final_test_ground_truth, final_test_prediction, final_test_probability), f)

        print("Saved Attack Test Ground Truth and Predict Sets")
        print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

    test_accuracy = 1.0 * correct / total
    print('Test Acc: %.3f%% (%d/%d)' % (100. * test_accuracy, correct, total))

    final_result = [test_f1_score, test_roc_auc_score, test_accuracy] if epoch else [test_accuracy]
    return final_result
