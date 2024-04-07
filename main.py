import argparse

from evaluate.kmeans_evaluate import KmeansDataset, attack_mode_0, evaluate_attack_model
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
        # 婚姻情况（Marital status）、性别（Gender）、入学年龄（Age at enrollment）、国籍（Nationality）、入学成绩（Admission grade)
        qid_indices_names = ["Marital status", "Gender", "Age at enrollment", "Nationality", "Admission grade"]
        qid_indices = [0, 17, 19, 7, 12]
        dataset = Student()

    elif dataset_name == 'Obesity':
        dataset = Obesity()
        # 性别（Gender）、年龄（Age）、身高（Height）和体重（Weight）
        qid_indices_names = ["Gender", "Age", "Height", "Weight"]
        qid_indices = [0, 1, 2, 3]

    elif dataset_name == 'Hospital':
        dataset = Hospital()
        # 性别（Gender）、年龄（Age）、人种（Race）和体重（Weight）
        qid_indices_names = ["Race", "Gender", "Age", "Weight"]
        qid_indices = [2, 3, 4, 5]

    elif dataset_name == 'Adult':
        dataset = Adult()
        # 年龄（Age）、人种（Race）、性别（Gender）、国家（Native-country）
        qid_indices_names = ["Age", "Race", "Gender", "Native-country"]
        qid_indices = [0, 8, 9, 13]

    evaluator = QID_VE(dataset)
    evaluator.train_test_split()

    impacts = []
    for i in qid_indices:
        impact = evaluator.compute_qid_impacts(i)
        print(f"Impact for QID at index {i}: {impact}")
        impacts.append(impact)

    # 将impact结果处理为和为一
    total_impact = sum(impacts)
    normalized_impacts = [impact / total_impact for impact in impacts]

    # 输出标准化后的impact值
    for index, normalized_impact in zip(range(len(qid_indices)), normalized_impacts):
        print(f"Normalized Impact for QID at {qid_indices_names[index]}: {normalized_impact}")


def test_kmeans(dataset_name, model_name, mode, train_target, train_shadow, device):
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
        num_classes = 3

    # 打印前两个样本
    for i in range(2):
        X, target = dataset[i]
        print(f'Sample {i}: {X}, Target: {target}')

    TARGET_PATH = "./dataloader/trained_model/" + dataset_name + model_name

    # 取前三分之一样本的数据
    # 这个数据和train_target_model中的batch_size有关
    n = 500

    # 获取三类数据集 min max random
    evaluator = KmeansDataset(dataset)
    min_dataset, max_dataset, random_dataset, random_dataset_shadow = evaluator.get_specific_datasets_and_distances(n)
    num_features = next(iter(dataset))[0].shape[0]
    each_length = n // 2

    target_train_min, target_test_min = torch.utils.data.random_split(
        min_dataset, [each_length, each_length]
    )
    target_train_max, target_test_max = torch.utils.data.random_split(
        max_dataset, [each_length, each_length]
    )
    target_train_random, target_test_random = torch.utils.data.random_split(
        random_dataset_shadow, [each_length, each_length]
    )

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
        # _random_target.pth
        train_target_model(TARGET_PATH + "_random", device, target_train_random, target_test_random, target_model,
                           model_name, num_features)
    if train_shadow:
        # _shadow.pth
        train_shadow_model(TARGET_PATH, device, shadow_train, shadow_test, shadow_model, model_name,
                           num_features)

    # 训练攻击模型+生成测试数据集
    test_mia(TARGET_PATH, device, num_classes, target_train_min, target_test_min,
             shadow_train, shadow_test,
             target_model, shadow_model, mode, model_name, num_features, "_min")
    test_mia(TARGET_PATH, device, num_classes, target_train_max, target_test_max,
             shadow_train, shadow_test,
             target_model, shadow_model, mode, model_name, num_features, "_max")
    test_mia(TARGET_PATH, device, num_classes, target_train_random, target_test_random,
             shadow_train, shadow_test,
             target_model, shadow_model, mode, model_name, num_features, "_random")

    attack_min_model_path = TARGET_PATH + '_min' + '_meminf_attack0.pth'
    attack_max_model_path = TARGET_PATH + '_max' + '_meminf_attack0.pth'
    attack_random_model_path = TARGET_PATH + '_random' + '_meminf_attack0.pth'

    test_min_set_path = TARGET_PATH + '_min' + '_meminf_attack_mode0_test.p'
    test_max_set_path = TARGET_PATH + '_max' + '_meminf_attack_mode0_test.p'
    test_random_set_path = TARGET_PATH + '_random' + '_meminf_attack_mode0_test.p'

    attack_model_path = attack_random_model_path
    result_path = './dataloader/trained_model/attack_results.p'
    print("========MIN_dataset========")
    evaluate_attack_model(attack_model_path, test_min_set_path, result_path, num_classes, 1)
    print("========MAX_dataset========")
    evaluate_attack_model(attack_model_path, test_max_set_path, result_path, num_classes, 1)
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
        attack_mode0(PATH + kmeans_mode + "_target.pth", PATH + "_shadow.pth", PATH + kmeans_mode, device,
                     attack_trainloader,
                     attack_testloader,
                     target_model, shadow_model, attack_model, 1, model_name, num_features)
    # 进行MIA评估 黑盒+Partial辅助数据集
    elif mode == 1:
        attack_model = PartialAttackModel(num_classes)
        attack_mode1(PATH + "_target.pth", PATH, device, attack_trainloader, attack_testloader,
                     target_model,
                     attack_model, 1, model_name, num_features)


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

    # 打印前两个样本
    for i in range(2):
        X, target = dataset[i]
        print(f'Sample {i}: {X}, Target: {target}')

    # 划分目标和影子数据集
    length = len(dataset)
    each_train_length = length // 4
    each_test_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_train_length, each_test_length, each_train_length, each_test_length,
                  length - (each_train_length * 2 + each_test_length * 2)]
    )
    num_features = next(iter(dataset))[0].shape[0]

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
    # 进行kmeans聚类研究
    elif args.evaluate_type == 1:
        test_kmeans(dataset_name, model_name, mode, args.train_target, args.train_shadow, device)
    # 进行QID脆弱性研究
    elif args.evaluate_type == 2:
        test_QID(dataset_name)


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
