import argparse

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
        print('Obesity_kmeans')
        dataset = Obesity()
        num_classes = 7
    elif dataset_name == 'Student':
        print('Student_kmeans')
        dataset = Student()
        num_classes = 3
    elif dataset_name == 'Hospital':
        print('Hospital_kmeans')
        dataset = Hospital()
        num_classes = 3
    elif dataset_name == 'Adult':
        print('Adult_kmeans')
        dataset = Adult()
        num_classes = 2

    # 打印前两个样本
    for i in range(2):
        X, target = dataset[i]
        print(f'Sample {i}: {X}, Target: {target}')

    TARGET_PATH = "./dataloader/trained_model/" + dataset_name + model_name

    dataset_len = len(dataset)
    # 取前三分之一样本的数据
    # 这个数据和train_target_model中的batch_size有关
    n = dataset_len // 7
    if n >= 2000:
        n = 2000

    # 获取三类数据集 min max random
    evaluator = HDBSCANDataset(dataset)
    min_dataset, max_dataset, noise_dataset, random_dataset, test_dataset, random_dataset_shadow = evaluator.get_specific_datasets_and_distances(
        n)
    num_features = next(iter(dataset))[0].shape[0]
    each_length = n

    # target_train_min, target_test_min = torch.utils.data.random_split(
    #     min_dataset, [each_length, each_length]
    # )
    # target_train_max, target_test_max = torch.utils.data.random_split(
    #     max_dataset, [each_length, each_length]
    # )
    # target_train_random, target_test_random = torch.utils.data.random_split(
    #     random_dataset, [each_length, each_length]
    # )

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

    result_path = './dataloader/trained_model/attack_results.p'
    print("========MIN_dataset========")
    evaluate_attack_model(attack_model_path, test_min_set_path, result_path, num_classes, 1)
    print("========MAX_dataset========")
    evaluate_attack_model(attack_model_path, test_max_set_path, result_path, num_classes, 1)
    print("========NOISE_dataset========")
    evaluate_attack_model(attack_model_path, test_noise_set_path, result_path, num_classes, 1)
    print("========RANDOM_dataset========")
    evaluate_attack_model(attack_model_path, test_random_set_path, result_path, num_classes, 1)


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

    # 训练目标模型
    if args.train_target and not args.kmeans:
        train_target_model(TARGET_PATH, device, target_train, target_test, target_model, model_name, num_features)

    # 训练影子模型
    if args.train_shadow and not args.kmeans:
        train_shadow_model(TARGET_PATH, device, shadow_train, shadow_test, shadow_model, model_name, num_features)

    # ----- 进行隐私风险评估 ----- #
    # 进行MIA评估
    if args.evaluate_type == 0:
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
        normalized_impacts = test_QID(dataset_name)  # 假设这个函数返回每个QID的影响力评分列表

        # 计算QID影响力评分的平均值和标准差
        mean_impact = sum(normalized_impacts) / len(normalized_impacts)
        std_impact = (sum((x - mean_impact) ** 2 for x in normalized_impacts) / len(normalized_impacts)) ** 0.5

        # 计算变异系数
        print(f"QID总风险的标准差: {std_impact}")

        # ====MIA分析==== #
        print("开始MIA分析...")
        test_acc = test_mia(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test,
                 target_model, shadow_model, mode, model_name, num_features)

        print(f"MIA攻击准确率为: {test_acc}")

        comprehensive_privacy_risk_score = (1 - test_acc) * (1 + std_impact)
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
