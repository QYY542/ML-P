import argparse

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from evaluate.hdbscan_evaluate import HDBSCANDataset, evaluate_attack_model, test_hdbscan_mia, get_attack_dataset_with_shadow_hdbscan
from evaluate.mia_evaluate import *
from dataloader.dataloader_adult import Adult
from dataloader.dataloader_attack import get_attack_dataset_with_shadow, get_attack_dataset_without_shadow
from dataloader.dataloader_hospital import Hospital
from dataloader.dataloader_obesity import Obesity
from dataloader.dataloader_student import Student
from evaluate.qid_evaluate import QID_VE
from models.train_models import *
from models.define_models import *
from dataloader.dataloader import *


def test_QID(dataset_name):
    if dataset_name == 'Student':
        # 婚姻情况（Marital status）、性别（Gender）、入学年龄（Age at enrollment）、国籍（Nationality）
        qid_indices_names = ["Marital status", "Nationality", "Gender", "Age at enrollment"]
        qid_indices = [0, 7, 17, 19]
        dataset = Student(qid_indices=qid_indices)

    elif dataset_name == 'Obesity':
        # 性别（Gender）、年龄（Age）、身高（Height）和体重（Weight）
        qid_indices_names = ["Gender", "Age", "Height", "Weight"]
        qid_indices = [0, 1, 2, 3]
        dataset = Obesity(qid_indices=qid_indices)

    elif dataset_name == 'Hospital':
        # 性别（Gender）、年龄（Age）、人种（Race）和体重（Weight）
        qid_indices_names = ["Race", "Gender", "Age", "Weight"]
        qid_indices = [2, 3, 4, 5]
        dataset = Hospital(qid_indices=qid_indices)

    elif dataset_name == 'Adult':
        # 年龄（Age）、人种（Race）、性别（Gender）、国家（Native-country）
        qid_indices_names = ["Age", "Race", "Gender", "Native-country"]
        qid_indices = [0, 8, 9, 13]
        dataset = Adult(qid_indices=qid_indices)

    # 打印前两个样本
    for i in range(2):
        X, target = dataset[i]
        print(f'Sample {i}: {X}, Target: {target}')

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

    return normalized_impacts


def test_hdbscan(dataset_name, model_name, mode, train_target, train_shadow, device):
    # 假设您已经正确加载了数据集
    if dataset_name == 'Obesity':
        print('Obesity_hdbscan')
        dataset = Obesity()
        num_classes = 7
    elif dataset_name == 'Student':
        print('Student_hdbscan')
        dataset = Student()
        num_classes = 3
    elif dataset_name == 'Hospital':
        print('Hospital_hdbscan')
        dataset = Hospital()
        num_classes = 3
    elif dataset_name == 'Adult':
        print('Adult_hdbscan')
        dataset = Adult()
        num_classes = 2

    # 打印前两个样本
    for i in range(2):
        X, target = dataset[i]
        print(f'Sample {i}: {X}, Target: {target}')

    TARGET_PATH = "./dataloader/trained_model/" + dataset_name + model_name

    # 取400份样本的数据
    dataset_len = len(dataset)
    n = 400

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

    draw(result_path,dataset_name, model_name)
    draw_tpr(result_path,dataset_name, model_name)

    return test_acc_min, test_acc_max, test_acc_noise, test_acc_random, distances


def draw(result_path,dataset_name, model_name):
    result_paths = [
        result_path + "_min.p",
        result_path + "_max.p",
        result_path + "_noise.p",
        result_path + "_random.p"
    ]
    datasets = ["MIN", "MAX", "NOISE", "RANDOM"]
    correct_probabilities = []
    incorrect_probabilities = []
    means_correct = []
    means_incorrect = []

    for path in result_paths:
        with open(path, "rb") as f:
            final_train_gndtrth, final_train_predict, final_train_probabe = pickle.load(f)

        # 筛选出预测正确的样本的预测概率
        correct_probs = [prob for prob, predict, real in
                         zip(final_train_probabe, final_train_predict, final_train_gndtrth) if predict == real]
        correct_probabilities.append(correct_probs)
        means_correct.append(np.mean(correct_probs))

        # 筛选出预测错误的样本的预测概率
        incorrect_probs = [prob for prob, predict, real in
                           zip(final_train_probabe, final_train_predict, final_train_gndtrth) if predict != real]
        incorrect_probabilities.append(incorrect_probs)
        means_incorrect.append(np.mean(incorrect_probs) if incorrect_probs else 0)

    # 绘制预测概率的分布图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 为每个数据集绘制预测正确的样本的预测概率分布
    parts_correct = ax.violinplot(correct_probabilities, positions=np.arange(1, len(datasets) * 2, 2) - 0.15,
                                  showmeans=False, showmedians=False, showextrema=False)
    for pc in parts_correct['bodies']:
        pc.set_facecolor('#DDA0DD')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # 为每个数据集绘制预测错误的样本的预测概率分布
    parts_incorrect = ax.violinplot(incorrect_probabilities, positions=np.arange(1, len(datasets) * 2, 2) + 0.15,
                                    showmeans=False, showmedians=False, showextrema=False)
    for pc in parts_incorrect['bodies']:
        pc.set_facecolor('#1E90FF')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # 标出平均值
    ax.scatter(np.arange(1, len(datasets) * 2, 2) - 0.15, means_correct, marker='o', color='red', s=30, zorder=3,
               label='Mean Probability (Correct)')
    ax.scatter(np.arange(1, len(datasets) * 2, 2) + 0.15, means_incorrect, marker='o', color='blue', s=30, zorder=3,
               label='Mean Probability (Incorrect)')

    # 添加一些图形属性来增加可读性
    ax.set_title('Predicted Probability Distribution Across Datasets')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Predicted Probability')
    ax.set_xticks(np.arange(1, len(datasets) * 2, 2))
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.savefig("./dataloader/" + dataset_name + "_" + model_name + "_predicted_probability_distribution.png")


def draw_tpr(result_path, dataset_name, model_name):
    result_paths = [
        result_path + "_min.p",
        result_path + "_max.p",
        result_path + "_noise.p",
        result_path + "_random.p"
    ]
    datasets = ["MIN", "MAX", "NOISE", "RANDOM"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, path in enumerate(result_paths):
        with open(path, "rb") as f:
            final_train_gndtrth, final_train_predict, final_train_probabe = pickle.load(f)

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(final_train_gndtrth, final_train_probabe)

        # 绘制ROC曲线
        ax.plot(fpr, tpr, label=f"{datasets[i]} (area = {np.trapz(tpr, fpr):.2f})", lw=2)

    # 设置图表属性
    ax.set_title('ROC Curve on Logarithmic Scale')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-2, 1e0])
    ax.set_ylim([1e-2, 1e0])
    ax.legend(loc='best')

    # 添加网格
    ax.grid(True, which="both", linestyle='--')

    # 保存并显示图表
    plt.savefig("./dataloader/" + dataset_name + "_" + model_name + "_roc_curve_log_scale.png")
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
    if dataset_name == "Adult":
        print("Adult")
        dataset = Adult()
        num_classes = 2
    elif dataset_name == "Obesity":
        print("Obesity")
        dataset = Obesity()
        num_classes = 7
    elif dataset_name == "Student":
        print("Student")
        dataset = Student()
        num_classes = 3
    elif dataset_name == "Hospital":
        print("Hospital")
        dataset = Hospital()
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
        # ====QID脆弱性分析==== #
        print("开始QID脆弱性分析...")
        normalized_impacts = test_QID(dataset_name)
        mean_impact = sum(normalized_impacts) / len(normalized_impacts)
        std_impact = (sum((x - mean_impact) ** 2 for x in normalized_impacts) / len(normalized_impacts)) ** 0.5

        # ====HDBSCAN聚类分析==== #
        print("开始HDBSCAN聚类分析...")
        test_acc_min, test_acc_max, test_acc_noise, test_acc_random, distances = test_hdbscan(dataset_name, model_name, mode, args.train_target, args.train_shadow, device)
        cluster_attack_success_rates = [test_acc_min, test_acc_max, test_acc_noise, test_acc_random]
        mean_cluster_success_rate = sum(cluster_attack_success_rates) / len(cluster_attack_success_rates)
        std_cluster_success_rate = (sum(
            (x - mean_cluster_success_rate) ** 2 for x in cluster_attack_success_rates) / len(
            cluster_attack_success_rates)) ** 0.5

        # ====MIA分析==== #
        print("开始MIA分析...")
        if args.train_target:
            train_target_model(TARGET_PATH, device, target_train, target_test, target_model, model_name, num_features)

        # 训练影子模型
        if args.train_shadow:
            train_shadow_model(TARGET_PATH, device, shadow_train, shadow_test, shadow_model, model_name, num_features)

        test_acc = test_mia(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test,target_model, shadow_model, mode, model_name, num_features)

        result_path = TARGET_PATH + "_meminf_attack0.p"
        # 打开并读取文件内容
        with open(result_path, "rb") as f:
            final_train_gndtrth, final_train_predict, final_train_probabe = pickle.load(f)

        # 找到预测标签与真实标签一致的样本的索引
        correct_predictions_indices = [i for i, (predict, real) in
                                       enumerate(zip(final_train_predict, final_train_gndtrth)) if predict == real]

        # 提取预测正确的样本的预测概率
        correct_probabilities = [final_train_probabe[i] for i in correct_predictions_indices]

        # 根据预测概率排序，找到预测概率最高的前10个样本的索引（在预测正确的子集中）
        top_10_highest_prob_indices = sorted(range(len(correct_probabilities)), key=lambda i: correct_probabilities[i],
                                             reverse=True)[:10]

        # 从原始的样本集中获取这10个样本的详细信息
        top_10_highest_prob_samples = [(correct_predictions_indices[i],
                                        final_train_gndtrth[correct_predictions_indices[i]],
                                        final_train_predict[correct_predictions_indices[i]],
                                        final_train_probabe[correct_predictions_indices[i]]) for i in
                                       top_10_highest_prob_indices]

        for i, (index, gndtrth, predict, probabe) in enumerate(top_10_highest_prob_samples, 1):
            print(f"样本 {i}: 索引 {index}, 真实标签 {gndtrth}, 预测标签 {predict}, 预测概率 {probabe}")

        # 总结
        print("test_acc_min, test_acc_max, test_acc_noise, test_acc_random", test_acc_min, test_acc_max, test_acc_noise, test_acc_random)
        print(f"聚类总风险的标准差: {std_cluster_success_rate}")
        print(f"QID总风险的标准差: {std_impact}")
        print(f"MIA攻击准确率为: {test_acc}")

        comprehensive_privacy_risk_score = test_acc * (1 + std_impact) * (1 + std_cluster_success_rate)
        print(f"数据集的综合隐私评分为: {comprehensive_privacy_risk_score}")



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
