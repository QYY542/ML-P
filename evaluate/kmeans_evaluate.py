import pickle
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Subset

from dataloader.dataloader_attack import get_attack_dataset_with_shadow, get_attack_dataset_without_shadow
from evaluate.mia_evaluate import attack_for_blackbox, attack_mode0, attack_mode1
from models.define_models import ShadowAttackModel, PartialAttackModel


class KmeansDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def compute_kmeans_distance(self):
        n_clusters = self.elbow_method()  # 使用肘部方法确定最佳聚类数
        print("n_clusters: ", n_clusters)
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
        random_shadow_indices = np.random.choice(len(self.dataset), n+n, replace=False)

        # 获取对应的聚类距离
        min_distances_values = min_distances[min_indices]
        max_distances_values = min_distances[max_indices]

        # 打印聚类距离
        print("聚类距离最小的样本距离:", min_distances_values)
        print("聚类距离最大的样本距离:", max_distances_values)

        # 根据索引创建数据子集
        min_dataset = Subset(self.dataset, min_indices)
        max_dataset = Subset(self.dataset, max_indices)
        random_dataset = Subset(self.dataset, random_indices)
        random_dataset_shadow = Subset(self.dataset, random_shadow_indices)

        # 排除min_dataset, max_dataset, random_dataset的索引
        excluded_indices = set(min_indices).union(max_indices, random_indices)
        all_indices = set(range(len(self.dataset)))
        remaining_indices = list(all_indices - excluded_indices)

        # 从剩余索引中选择随机数据集作为test_dataset
        test_indices = np.random.choice(remaining_indices, n, replace=False)
        test_dataset = Subset(self.dataset, test_indices)

        return min_dataset, max_dataset, random_dataset, test_dataset, random_dataset_shadow

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
        sse_array = np.array(sse)  # 将sse列表转换为NumPy数组
        # 将SSE数组标准化到0到1之间
        sse_normalized = (sse_array - np.min(sse_array)) / (np.max(sse_array) - np.min(sse_array))
        n_points = len(sse_normalized)
        all_coords = np.vstack((np.arange(n_points), sse_normalized)).T
        # 第一个点和最后一个点
        first_point = all_coords[0]
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
        # 计算每一个点到直线的距离
        distances = []
        for point in all_coords:
            vec_from_first = point - first_point
            cross_vec = np.cross(line_vec_norm, vec_from_first)
            dist = np.linalg.norm(cross_vec)
            distances.append(dist)
        # 找到距离最大的点作为肘部点
        elbow_index = np.argmax(distances)
        return elbow_index + 1  # 由于聚类数不能为0，所以加1



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

def test_kmeans_mia(PATH, device, num_classes, attack_trainloader, attack_testloader, target_model,
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

def get_attack_dataset_with_shadow_kmeans(target_train_min, target_test_min, target_train_max, target_test_max, target_train_random, target_test_random, shadow_train, shadow_test, batch_size = 64):
    mem_train, nonmem_train, mem_test_min, nonmem_test_min, mem_test_max, nonmem_test_max, mem_test_random, nonmem_test_random = list(shadow_train), list(shadow_test), list(target_train_min), list(
        target_test_min), list(target_train_max), list(target_test_max), list(target_train_random), list(target_test_random)

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

    for i in range(len(nonmem_test_random)):
        nonmem_test_random[i] = nonmem_test_random[i] + (0,)
    for i in range(len(mem_test_random)):
        mem_test_random[i] = mem_test_random[i] + (1,)

    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test_min), len(nonmem_test_min))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])

    mem_test_min, _ = torch.utils.data.random_split(mem_test_min, [test_length, len(mem_test_min) - test_length])
    non_mem_test_min, _ = torch.utils.data.random_split(nonmem_test_min, [test_length, len(nonmem_test_min) - test_length])

    mem_test_max, _ = torch.utils.data.random_split(mem_test_max, [test_length, len(mem_test_max) - test_length])
    non_mem_test_max, _ = torch.utils.data.random_split(nonmem_test_max, [test_length, len(nonmem_test_max) - test_length])

    mem_test_random, _ = torch.utils.data.random_split(mem_test_random, [test_length, len(mem_test_random) - test_length])
    non_mem_test_random, _ = torch.utils.data.random_split(nonmem_test_random, [test_length, len(nonmem_test_random) - test_length])

    attack_train = mem_train + non_mem_train
    attack_test_min = mem_test_min + non_mem_test_min
    attack_test_max = mem_test_max + non_mem_test_max
    attack_test_random = mem_test_random + non_mem_test_random

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_min_testloader = torch.utils.data.DataLoader(
        attack_test_min, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_max_testloader = torch.utils.data.DataLoader(
        attack_test_max, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_random_testloader = torch.utils.data.DataLoader(
        attack_test_random, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_min_testloader, attack_max_testloader, attack_random_testloader