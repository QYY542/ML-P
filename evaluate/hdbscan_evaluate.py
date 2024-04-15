from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Subset
import numpy as np
import hdbscan
import pickle
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from dataloader.dataloader_attack import get_attack_dataset_with_shadow, get_attack_dataset_without_shadow
from evaluate.mia_evaluate import attack_for_blackbox, attack_mode0, attack_mode1
from models.define_models import ShadowAttackModel, PartialAttackModel


# 添加噪音点的数据集训练出来的数据集隐私风险低
class HDBSCANDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.min_cluster_size = 3

    def load_and_scale_data(self):
        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for X, _ in loader:
            X = X.numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def compute_hdbscan_clusters(self):
        X_scaled = self.load_and_scale_data()
        X_scaled = X_scaled.astype(np.float64)
        print("min_cluster_size = ", self.min_cluster_size)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, gen_min_span_tree=True, metric='manhattan')
        clusterer.fit(X_scaled)
        return clusterer.labels_, X_scaled, clusterer.probabilities_

    def get_distances_and_probabilities(self, labels, X_scaled, probabilities):
        unique_labels = np.unique(labels)

        # 计算每个簇的中心点
        cluster_centers = {label: X_scaled[labels == label].mean(axis=0) for label in unique_labels if label != -1}

        # 计算全局中心点
        global_center = np.mean(X_scaled, axis=0) if len(cluster_centers) > 0 else np.zeros(X_scaled.shape[1])

        # 距离计算，增加归属概率的影响
        distances = np.zeros(len(X_scaled))
        for i in range(X_scaled.shape[0]):
            if labels[i] != -1:
                # 簇内的点
                cluster_density = len(X_scaled[labels == labels[i]]) / np.mean(
                    np.linalg.norm(X_scaled[labels == labels[i]] - cluster_centers[labels[i]], axis=1))
                base_distance = np.sum(np.abs(X_scaled[i] - global_center))
                adjusted_distance = (base_distance * (1.5 - probabilities[i])) / cluster_density
                distances[i] = adjusted_distance
            else:
                # 噪声点，赋予更高的基础距离
                distances[i] = np.sum(np.abs(X_scaled[i] - global_center)) * 1.5

        return distances

    # def visualize_clusters(self, X_scaled, labels):
    #     # 将数据降维到二维
    #     pca = PCA(n_components=3)
    #     X_pca = pca.fit_transform(X_scaled)
    #
    #     plt.figure(figsize=(12, 8))
    #     unique_labels = np.unique(labels)
    #     colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    #
    #     for k, col in zip(unique_labels, colors):
    #         if k == -1:
    #             # 黑色用于噪声点
    #             col = 'k'
    #
    #         class_member_mask = (labels == k)
    #
    #         xy = X_pca[class_member_mask]
    #         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
    #                  markeredgecolor='k', markersize=6 if k == -1 else 10,
    #                  alpha=0.6 if k == -1 else 0.8)
    #
    #     plt.title('Clustered data by HDBSCAN (PCA Reduced)')
    #     plt.xlabel('PCA Component 1')
    #     plt.ylabel('PCA Component 2')
    #     plt.grid(True)
    #     plt.show()


    def get_specific_datasets_and_distances(self, n):
        labels, X_scaled, probabilities = self.compute_hdbscan_clusters()
        distances = self.get_distances_and_probabilities(labels, X_scaled, probabilities)

        # 找出非噪声点的索引
        non_noise_indices = np.where(labels != -2)[0]
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
            # selected_noise_indices = noise_sorted_indices[-min(len(noise_sorted_indices), n):]
            selected_noise_distances = noise_distances[noise_sorted_indices_tmp[:min(len(noise_sorted_indices), n)]]
            # selected_noise_distances = noise_distances[noise_sorted_indices_tmp[-min(len(noise_sorted_indices), n):]]
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

        without_low_distance_indices = [index for index in range(len(self.dataset)) if
                                        index not in low_distance_indices]
        without_low_distance_dataset = Subset(self.dataset, without_low_distance_indices)


        print("簇内聚类距离小的样本分数:", low_distance_values)
        print("簇内聚类距离大的样本分数:", high_distance_values)
        print("噪音点距离最大的样本分数：", selected_noise_distances)
        print("聚类距离随机的样本分数:", random_shadow_values)
        print(len(noise_indices))

        return low_distance_dataset, high_distance_dataset, noise_dataset, random_dataset, test_dataset, random_shadow_dataset, distances


def evaluate_attack_model(model_path, test_set_path, result_path, num_classes):
    # 加载攻击模型
    attack_model = ShadowAttackModel(num_classes)
    attack_model.load_state_dict(torch.load(model_path, map_location=attack_model.device))
    attack_model.eval()

    correct = 0
    total = 0
    final_test_ground_truth = []
    final_test_prediction = []
    final_test_probability = []
    # 新增统计计数器
    predicted_members_correct = 0
    predicted_members_total = 0

    with torch.no_grad():
        with open(test_set_path, "rb") as f:
            while True:
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(attack_model.device), prediction.to(
                        attack_model.device), members.to(attack_model.device)

                    results = attack_model(output, prediction)
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    # 更新新增计数器
                    predicted_members = predicted == 1
                    predicted_members_total += predicted_members.sum().item()
                    predicted_members_correct += (predicted == members).logical_and(predicted == 1).sum().item()

                    probabilities = F.softmax(results, dim=1)
                    final_test_ground_truth.append(members)
                    final_test_prediction.append(predicted)
                    final_test_probability.append(probabilities[:, 1])
                except EOFError:
                    break

    final_test_ground_truth = torch.cat(final_test_ground_truth, dim=0).cpu().numpy()
    final_test_prediction = torch.cat(final_test_prediction, dim=0).cpu().numpy()
    final_test_probability = torch.cat(final_test_probability, dim=0).cpu().numpy()

    with open(result_path, "wb") as f:
        pickle.dump((final_test_ground_truth, final_test_prediction, final_test_probability), f)

    print("Saved Attack Test Ground Truth and Predict Sets")
    # 输出正确且为成员的样本与预测为成员的样本的比值
    if predicted_members_total > 0:
        precision = predicted_members_correct / predicted_members_total
        print("Precision of Correct Member Predictions: %.3f" % precision)
    else:
        print("No member predictions were made.")

    test_acc = 1.0 * correct / total
    print('Test Acc: %.3f%% (%d/%d)' % (100. * test_acc, correct, total))

    return test_acc


def test_hdbscan_mia(PATH, device, num_classes, attack_trainloader, attack_testloader, target_model,
                     shadow_model, mode, model_name, num_features, kmeans_mode=""):
    # 进行MIA评估 黑盒+Shadow辅助数据集
    if mode == 0:
        attack_model = ShadowAttackModel(num_classes)
        attack_mode0(PATH + kmeans_mode + "_target.pth", PATH + "_shadow.pth", PATH + kmeans_mode, device,
                     attack_trainloader,
                     attack_testloader,
                     target_model, shadow_model, attack_model, 1, model_name, num_features)
    # 进行MIA评估 黑盒+Partial辅助数据集
    elif mode == 1:
        attack_model = PartialAttackModel(num_classes)
        attack_mode1(PATH + kmeans_mode + "_target.pth", PATH + kmeans_mode, device, attack_trainloader,
                     attack_testloader,
                     target_model,
                     attack_model, 1, model_name, num_features)


def get_attack_dataset_with_shadow_hdbscan(target_train_min, target_test_min, target_train_max, target_test_max,
                                           target_train_noise, target_test_noise, target_train_random,
                                           target_test_random, shadow_train, shadow_test, batch_size=64):
    mem_train, nonmem_train, mem_test_min, nonmem_test_min, mem_test_max, nonmem_test_max, mem_test_noise, nonmem_test_noise, mem_test_random, nonmem_test_random = list(
        shadow_train), list(shadow_test), list(target_train_min), list(
        target_test_min), list(target_train_max), list(target_test_max), list(target_train_noise), list(
        target_test_noise), list(target_train_random), list(
        target_test_random)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)

    for i in range(len(nonmem_test_min)):
        nonmem_test_min[i] = nonmem_test_min[i] + (0,)
    for i in range(len(mem_test_min)):
        mem_test_min[i] = mem_test_min[i] + (1,)

    for i in range(len(nonmem_test_max)):
        nonmem_test_max[i] = nonmem_test_max[i] + (0,)
    for i in range(len(mem_test_max)):
        mem_test_max[i] = mem_test_max[i] + (1,)

    for i in range(len(nonmem_test_noise)):
        nonmem_test_noise[i] = nonmem_test_noise[i] + (0,)
    for i in range(len(mem_test_noise)):
        mem_test_noise[i] = mem_test_noise[i] + (1,)

    for i in range(len(nonmem_test_random)):
        nonmem_test_random[i] = nonmem_test_random[i] + (0,)
    for i in range(len(mem_test_random)):
        mem_test_random[i] = mem_test_random[i] + (1,)

    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test_min), len(nonmem_test_min))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])

    mem_test_min, _ = torch.utils.data.random_split(mem_test_min, [test_length, len(mem_test_min) - test_length])
    non_mem_test_min, _ = torch.utils.data.random_split(nonmem_test_min,
                                                        [test_length, len(nonmem_test_min) - test_length])

    mem_test_max, _ = torch.utils.data.random_split(mem_test_max, [test_length, len(mem_test_max) - test_length])
    non_mem_test_max, _ = torch.utils.data.random_split(nonmem_test_max,
                                                        [test_length, len(nonmem_test_max) - test_length])

    mem_test_noise, _ = torch.utils.data.random_split(mem_test_noise, [test_length, len(mem_test_noise) - test_length])
    non_mem_test_noise, _ = torch.utils.data.random_split(nonmem_test_noise,
                                                          [test_length, len(nonmem_test_noise) - test_length])

    mem_test_random, _ = torch.utils.data.random_split(mem_test_random,
                                                       [test_length, len(mem_test_random) - test_length])
    non_mem_test_random, _ = torch.utils.data.random_split(nonmem_test_random,
                                                           [test_length, len(nonmem_test_random) - test_length])

    attack_train = mem_train + non_mem_train
    attack_test_min = mem_test_min + non_mem_test_min
    attack_test_max = mem_test_max + non_mem_test_max
    attack_test_noise = mem_test_noise + non_mem_test_noise
    attack_test_random = mem_test_random + non_mem_test_random

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_min_testloader = torch.utils.data.DataLoader(
        attack_test_min, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_max_testloader = torch.utils.data.DataLoader(
        attack_test_max, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_noise_testloader = torch.utils.data.DataLoader(
        attack_test_noise, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_random_testloader = torch.utils.data.DataLoader(
        attack_test_random, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_min_testloader, attack_max_testloader, attack_noise_testloader, attack_random_testloader