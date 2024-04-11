from torch.utils.data import DataLoader
# import hdbscan
import pickle
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import  MinMaxScaler
import torch
from torch.utils.data import DataLoader, Subset
from evaluate.mia_evaluate import attack_mode0, attack_mode1
from models.define_models import ShadowAttackModel, PartialAttackModel
from sklearn.cluster import HDBSCAN
from scipy.spatial import distance
# 添加噪音点的数据集训练出来的数据集隐私风险低
class HDBSCANDataset:
    def __init__(self, dataset):
        self.dataset = dataset
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
        print("min_cluster_size = ", self.min_cluster_size)

        # 使用HDBSCAN计算余弦距离聚类
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size)
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
                # 计算余弦距离，并应用距离调整因子
                adjusted_distances = [distance.cosine(cp, center) for idx, cp in enumerate(cluster_points)]
                distances[labels == label] = adjusted_distances
            else:
                noise_indices = np.where(labels == -1)[0]
                print(len(noise_indices))
                for index in noise_indices:
                    noise_point = X_scaled[index]
                    distances_to_centers = [distance.cosine(noise_point, center) for center in cluster_centers.values()]
                    # 对噪声点也应用距离调整因子
                    distances[index] = min(distances_to_centers)

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

        print("簇内聚类距离小的样本分数:", low_distance_values)
        print("簇内聚类距离大的样本分数:", high_distance_values)
        print("噪音点距离最大的样本分数：", selected_noise_distances)
        print("聚类距离随机的样本分数:", random_shadow_values)

        return low_distance_dataset, high_distance_dataset, noise_dataset, random_dataset, test_dataset, random_shadow_dataset, distances

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

    test_acc = 1.0 * correct / total
    print('Test Acc: %.3f%% (%d/%d)' % (100. * test_acc, correct, total))

    final_result = [test_f1_score, test_roc_auc_score, test_acc] if epoch else [test_acc]
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
    mem_train, nonmem_train, mem_test_min, nonmem_test_min, mem_test_max, nonmem_test_max,mem_test_noise, nonmem_test_noise, mem_test_random, nonmem_test_random = list(
        shadow_train), list(shadow_test), list(target_train_min), list(
        target_test_min), list(target_train_max), list(target_test_max),list(target_train_noise), list(target_test_noise), list(target_train_random), list(
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