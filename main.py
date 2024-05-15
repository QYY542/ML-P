import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from evaluate.hdbscan_evaluate import HDBSCANDataset, evaluate_attack_model, test_hdbscan_mia, get_attack_dataset_with_shadow_hdbscan
from evaluate.mia_evaluate import *
from dataloader.dataloader_adult import Adult
from dataloader.dataloader_attack import get_attack_dataset_with_shadow, get_attack_dataset_without_shadow
from dataloader.dataloader_obesity import Obesity
from dataloader.dataloader_student import Student
from evaluate.qid_evaluate import QID_VE
from models.train_models import *
from models.define_models import *
import matplotlib.pyplot as plt
import csv

def test_QID(dataset_name):
    if dataset_name == "student" or dataset_name == "protected_student":
        DP = None
        if dataset_name == "protected_student":
            DP = [17, 19]
        # 婚姻情况（Marital status）、性别（Gender）、入学年龄（Age at enrollment）、国籍（Nationality）
        qid_indices_names = ["Marital status", "Nationality", "Gender", "Age at enrollment"]
        qid_indices = [0, 7, 17, 19]
        dataset = Student(filename = "student", qid_indices=qid_indices, DP_indices=DP)

    elif dataset_name == "obesity" or dataset_name == "protected_obesity":
        DP = None
        if dataset_name == "protected_obesity":
            DP = [2, 3]
        # 性别（Gender）、年龄（Age）、身高（Height）和体重（Weight）
        qid_indices_names = ["Gender", "Age", "Height", "Weight"]
        qid_indices = [0, 1, 2, 3]
        dataset = Obesity(filename = "obesity", qid_indices=qid_indices, DP_indices=DP)

    elif dataset_name == "adult" or dataset_name == "protected_adult":
        DP = None
        if dataset_name == "protected_adult":
            DP = [0, 9]
        # 年龄（Age）、人种（Race）、性别（Gender）、国家（Native-country）
        qid_indices_names = ["Age", "Race", "Gender", "Native-country"]
        qid_indices = [0, 8, 9, 13]
        dataset = Adult(filename = "adult", qid_indices=qid_indices, DP_indices=DP)

    evaluator = QID_VE(dataset)
    evaluator.train_test_split()

    impacts = []
    for i in range(4):
        impact = evaluator.compute_qid_impacts(i)
        print(f"Impact for QID at index {i}: {impact}")
        impacts.append(impact)

    # 将impact结果处理为和为一
    total_impact = sum(impacts)
    normalized_impacts = [impact / total_impact for impact in impacts]

    # 输出标准化后的impact值
    for index, normalized_impact in zip(range(len(qid_indices)), normalized_impacts):
        print(f"Normalized Impact for QID at {qid_indices_names[index]}: {normalized_impact}")

    print("RF预测成功率:", evaluator.evaluate_model())

    return normalized_impacts


def test_hdbscan(dataset_name, model_name, mode, train_target, train_shadow, device):
    # 数据集
    if dataset_name == "adult" or dataset_name == "protected_adult":
        print("Adult")
        dataset = Adult(filename = dataset_name)
        num_classes = 2
    elif dataset_name == "obesity" or dataset_name == "protected_obesity":
        print("Obesity")
        dataset = Obesity(filename = dataset_name)
        num_classes = 7
    elif dataset_name == "student" or dataset_name == "protected_student":
        print("Student")
        dataset = Student(filename = dataset_name)
        num_classes = 3

    # 打印前两个样本
    for i in range(2):
        X, target = dataset[i]
        print(f'Sample {i}: {X}, Target: {target}')

    TARGET_PATH = "./dataloader/trained_model/" + dataset_name + model_name

    # 1/6份样本的数据
    dataset_len = len(dataset)
    n = dataset_len // 6

    # 获取三类数据集 min max random
    evaluator = HDBSCANDataset(dataset)
    min_dataset, max_dataset, noise_dataset, random_dataset, test_dataset, random_dataset_shadow, distances = evaluator.get_specific_datasets_and_distances(n)
    num_features = next(iter(dataset))[0].shape[0]
    each_length = n

    target_train_min, target_test_min = min_dataset, test_dataset
    target_train_max, target_test_max = max_dataset, test_dataset
    target_train_noise, target_test_noise = noise_dataset, test_dataset
    target_train_random, target_test_random = random_dataset, test_dataset

    shadow_train, shadow_test = torch.utils.data.random_split(
        random_dataset_shadow, [each_length, each_length]
    )

    # 获取模型并且评估
    if model_name == "MLP":
        print("MLP")
        target_model = MLP(num_features, num_classes)
        shadow_model = MLP(num_features, num_classes)
    elif model_name == "ResNet":
        print("ResNet")
        print(num_features)
        print(num_classes)
        target_model = ResNetModel(num_features, num_classes)
        shadow_model = ResNetModel(num_features, num_classes)

    if train_target:
        # _min_target.pth
        train_target_model(TARGET_PATH + "_min", device, target_train_min, target_test_min, target_model, model_name,
                           num_features)
        # _max_target.pth
        train_target_model(TARGET_PATH + "_max", device, target_train_max, target_test_max, target_model, model_name,
                           num_features)
        # _noise_target.pth
        train_target_model(TARGET_PATH + "_noise", device, target_train_noise, target_test_noise, target_model,
                           model_name,
                           num_features)
        # _random_target.pth
        train_target_model(TARGET_PATH + "_random", device, target_train_random, target_test_random, target_model,
                           model_name, num_features)
    if train_shadow:
        # _shadow.pth
        train_shadow_model(TARGET_PATH, device, shadow_train, shadow_test, shadow_model, model_name,
                           num_features)

    # 生成测试数据集+训练攻击模型
    attack_trainloader, attack_min_testloader, attack_max_testloader, attack_noise_testloader, attack_random_testloader = get_attack_dataset_with_shadow_hdbscan(
        target_train_min, target_test_min, target_train_max, target_test_max, target_train_noise, target_test_noise,
        target_train_random, target_test_random, shadow_train, shadow_test, batch_size=64)

    test_hdbscan_mia(TARGET_PATH, device, num_classes, attack_trainloader,
                     attack_min_testloader,
                     target_model, shadow_model, mode, model_name, num_features, "_min")

    test_hdbscan_mia(TARGET_PATH, device, num_classes, attack_trainloader,
                     attack_max_testloader,
                     target_model, shadow_model, mode, model_name, num_features, "_max")

    test_hdbscan_mia(TARGET_PATH, device, num_classes, attack_trainloader,
                     attack_noise_testloader,
                     target_model, shadow_model, mode, model_name, num_features, "_noise")

    test_hdbscan_mia(TARGET_PATH, device, num_classes, attack_trainloader,
                     attack_random_testloader,
                     target_model, shadow_model, mode, model_name, num_features, "_random")

    attack_model_path = ''
    test_min_set_path = ''
    test_max_set_path = ''
    test_noise_set_path = ''
    test_random_set_path = ''
    mode = 0
    if mode == 0:
        attack_model_path = TARGET_PATH + '_random' + '_meminf_attack0.pth'
        test_min_set_path = TARGET_PATH + '_min' + '_meminf_attack_mode0_test.p'
        test_max_set_path = TARGET_PATH + '_max' + '_meminf_attack_mode0_test.p'
        test_noise_set_path = TARGET_PATH + '_noise' + '_meminf_attack_mode0_test.p'
        test_random_set_path = TARGET_PATH + '_random' + '_meminf_attack_mode0_test.p'
    elif mode == 1:
        attack_model_path = TARGET_PATH + '_random' + '_meminf_attack1.pth'
        test_min_set_path = TARGET_PATH + '_min' + '_meminf_attack_mode1_test.p'
        test_max_set_path = TARGET_PATH + '_max' + '_meminf_attack_mode1_test.p'
        test_noise_set_path = TARGET_PATH + '_noise' + '_meminf_attack_mode1_test.p'
        test_random_set_path = TARGET_PATH + '_random' + '_meminf_attack_mode1_test.p'

    result_path = './dataloader/trained_model/attack_results'
    print("========MIN_dataset========")
    test_acc_min = evaluate_attack_model(attack_model_path, test_min_set_path, result_path + "_min.p", num_classes)
    print("========MAX_dataset========")
    test_acc_max = evaluate_attack_model(attack_model_path, test_max_set_path, result_path + "_max.p", num_classes)
    print("========NOISE_dataset========")
    test_acc_noise = evaluate_attack_model(attack_model_path, test_noise_set_path, result_path + "_noise.p", num_classes)
    print("========RANDOM_dataset========")
    test_acc_random = evaluate_attack_model(attack_model_path, test_random_set_path, result_path + "_random.p", num_classes)

    return test_acc_min, test_acc_max, test_acc_noise, test_acc_random, distances

def test_mia(PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model,
             shadow_model, mode, model_name, num_features, kmeans_mode=""):
    batch_size = 64

    # 获取攻击数据集
    if mode == 0:
        attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
            target_train, target_test, shadow_train, shadow_test, batch_size)
    else:
        attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)

    # 进行MIA评估 黑盒+Shadow辅助数据集
    if mode == 0:
        attack_model = ShadowAttackModel(num_classes)
        test_acc = attack_mode0(PATH + kmeans_mode + "_target.pth", PATH + "_shadow.pth", PATH + kmeans_mode, device,
                     attack_trainloader,
                     attack_testloader,
                     target_model, shadow_model, attack_model, 1, model_name, num_features)
    # 进行MIA评估 黑盒+Partial辅助数据集
    elif mode == 1:
        attack_model = PartialAttackModel(num_classes)
        test_acc = attack_mode1(PATH + kmeans_mode + "_target.pth", PATH + kmeans_mode, device, attack_trainloader,
                     attack_testloader,
                     target_model,
                     attack_model, 1, model_name, num_features)

    return test_acc


def prepare_dataset(dataset_name, model_name):
    # 数据集
    if dataset_name == "adult" or dataset_name == "protected_adult":
        print("Adult")
        dataset = Adult(filename = dataset_name)
        num_classes = 2
    elif dataset_name == "obesity" or dataset_name == "protected_obesity":
        print("Obesity")
        dataset = Obesity(filename = dataset_name)
        num_classes = 7
    elif dataset_name == "student" or dataset_name == "protected_student":
        print("Student")
        dataset = Student(filename = dataset_name)
        num_classes = 3

    # 划分目标和影子数据集
    length = len(dataset)
    each_train_length = length // 4
    each_test_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_train_length, each_test_length, each_train_length, each_test_length,
                  length - (each_train_length * 2 + each_test_length * 2)]
    )
    num_features = next(iter(dataset))[0].shape[0]

    target_model = []
    shadow_model = []
    # 模型
    if model_name == "MLP":
        print("MLP")
        target_model = MLP(num_features, num_classes)
        shadow_model = MLP(num_features, num_classes)
    elif model_name == "ResNet":
        print("ResNet")
        print(num_features)
        print(num_classes)
        target_model = ResNetModel(num_features, num_classes)
        shadow_model = ResNetModel(num_features, num_classes)

    return num_classes, num_features, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default="Net_1")
    parser.add_argument('--dataset', type=str, default="Student")
    parser.add_argument('--evaluate_type', type=int, default=0)
    parser.add_argument('--train_target', action='store_true')
    parser.add_argument('--train_shadow', action='store_true')
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--kmeans', action='store_true')
    args = parser.parse_args()

    # 处理从命令行获得的参数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")
    dataset_name = args.dataset
    model_name = args.model
    mode = args.mode
    TARGET_ROOT = "./dataloader/trained_model/"
    if not os.path.exists(TARGET_ROOT):
        print(f"Create directory named {TARGET_ROOT}")
        os.makedirs(TARGET_ROOT)
    TARGET_PATH = TARGET_ROOT + dataset_name + model_name

    # 获得目标数据集、影子数据集、目标模型、影子模型
    num_classes, num_features, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
        dataset_name, model_name)

    # ----- 进行隐私风险评估 ----- #
    # 进行MIA评估
    if args.evaluate_type == 0:
        # 训练目标模型
        if args.train_target:
            train_target_model(TARGET_PATH, device, target_train, target_test, target_model, model_name, num_features)

        # 训练影子模型
        if args.train_shadow:
            train_shadow_model(TARGET_PATH, device, shadow_train, shadow_test, shadow_model, model_name, num_features)

        test_mia(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test,
                 target_model, shadow_model, mode, model_name, num_features)

    # 进行HDBSCAN聚类研究
    elif args.evaluate_type == 1:
        test_hdbscan(dataset_name, model_name, mode, args.train_target, args.train_shadow, device)

    # 进行QID脆弱性研究
    elif args.evaluate_type == 2:
        test_QID(dataset_name)

    # 综合分析
    elif args.evaluate_type == 3:
        # 输出开始信息
        print("开始综合隐私风险分析...")

        # ====QID脆弱性分析==== #
        normalized_impacts = test_QID(dataset_name)
        mean_impact = sum(normalized_impacts) / len(normalized_impacts)
        std_QID = (sum((x - mean_impact) ** 2 for x in normalized_impacts) / len(normalized_impacts)) ** 0.5
        cv_QID = std_QID / mean_impact if mean_impact != 0 else 0  # 计算QID的变异系数

        # ====HDBSCAN聚类分析==== #
        test_acc_min, test_acc_max, test_acc_noise, test_acc_random, distances = test_hdbscan(dataset_name, model_name,
                                                                                              mode, args.train_target,
                                                                                              args.train_shadow, device)
        cluster_attack_success_rates = [test_acc_min, test_acc_max, test_acc_noise, test_acc_random]
        mean_cluster_success_rate = sum(cluster_attack_success_rates) / len(cluster_attack_success_rates)
        std_HDBSCAN = (sum((x - mean_cluster_success_rate) ** 2 for x in cluster_attack_success_rates) / len(
            cluster_attack_success_rates)) ** 0.5
        cv_HDBSCAN = std_HDBSCAN / mean_cluster_success_rate if mean_cluster_success_rate != 0 else 0  # 计算HDBSCAN的变异系数

        # ====MIA分析==== #
        if args.train_target:
            train_target_model(TARGET_PATH, device, target_train, target_test, target_model, model_name, num_features)
        if args.train_shadow:
            train_shadow_model(TARGET_PATH, device, shadow_train, shadow_test, shadow_model, model_name, num_features)
        test_acc = test_mia(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test,
                            target_model, shadow_model, mode, model_name, num_features)

        # 使用变异系数重新计算OPRS
        oprs_score = test_acc * (1 + cv_QID ) * (1 + cv_HDBSCAN)

        # 输出结果
        print(f"数据集的综合隐私评分 (OPRS) 为: {oprs_score:.4f}")


def fix_seed(num):
    # Set the random seed for NumPy
    np.random.seed(num)
    # Set the random seed for PyTorch
    torch.manual_seed(num)
    # If you are using CUDA (i.e., a GPU), also set the seed for it
    torch.cuda.manual_seed_all(num)


if __name__ == "__main__":
    fix_seed(43)
    main()
