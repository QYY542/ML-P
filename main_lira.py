import argparse
import math
from evaluate.hdbscan_evaluate import HDBSCANDataset, evaluate_attack_model, test_hdbscan_mia, get_attack_dataset_with_shadow_hdbscan
from evaluate.mia_evaluate import *
from dataloader.dataloader_adult import Adult
from dataloader.dataloader_attack import get_attack_dataset_with_shadow, get_attack_dataset_without_shadow
from dataloader.dataloader_obesity import Obesity
from dataloader.dataloader_student import Student
from evaluate.qid_evaluate import QID_VE
from models.train_models import *
from models.define_models import *
from lira.train import train  # 引入 train函数
from lira.inference import inference  # 引入 inference 函数
from lira.score import score # 引入 score 函数
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_dataset(evaluate_type, dataset_name, model_name):
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
    each_train_length = length // 2
    each_test_length = length // 2
    target_train, target_test, _ = torch.utils.data.random_split(
        dataset, [each_train_length, each_test_length,
                  length - (each_train_length + each_test_length)]
    )
    num_features = next(iter(dataset))[0].shape[0]
    print(f"length target_train:{len(target_train)}")
    print(f"length target_test:{len(target_test)}")

    target_model = []
    # 模型
    if model_name == "MLP":
        print("MLP")
        target_model = MLP(num_features, num_classes)
    elif model_name == "ResNet":
        print("ResNet")
        target_model = ResNetModel(num_features, num_classes)

    return num_classes, num_features, target_train, target_test, target_model

def test_MIA(if_train, DEVICE, model_name, TARGET_PATH, num_classes, num_features, target_train, target_test, target_model, evaluate_type, data_type):
    if if_train:
        train(TARGET_PATH, target_train, target_test, target_model, DEVICE, model_name, num_features, evaluate_type)
    inference(TARGET_PATH, target_train, target_test, DEVICE, model_name, num_features, num_classes, evaluate_type, data_type)
    roc_auc = score(TARGET_PATH, target_train, target_test, evaluate_type, data_type)
    print(f"roc_auc:{roc_auc}")
    return roc_auc

def test_HDBSCAN(args, dataset_name, model_name, target_train, target_test, target_model, DEVICE):
    # TODO 只训练一个模型，然后用这个模型来获取各个距离样本点的logits
    # 数据集
    if dataset_name == "adult" or dataset_name == "protected_adult":
        print("Adult")
        num_classes = 2
    elif dataset_name == "obesity" or dataset_name == "protected_obesity":
        print("Obesity")
        num_classes = 7
    elif dataset_name == "student" or dataset_name == "protected_student":
        print("Student")
        num_classes = 3
    TARGET_PATH = "./dataloader/trained_model/" + dataset_name + model_name
    
    dataset_len = len(target_train)
    num_features = next(iter(target_train))[0].shape[0]
    n = dataset_len // 4
    evaluator = HDBSCANDataset(target_train)
    min_dataset, max_dataset, noise_dataset, random_dataset = evaluator.get_specific_datasets_and_distances(n)
    print(f"{len(min_dataset)}:{len(max_dataset)}:{len(noise_dataset)}:{len(random_dataset)}")

    length = len(target_test)
    target_test, _ = torch.utils.data.random_split(target_test, [n,length - n])

    if_train = False
    roc_auc_min = test_MIA(if_train, DEVICE, model_name, TARGET_PATH, num_classes, num_features, min_dataset, target_test, target_model, "hdbscan", "min")
    roc_auc_max = test_MIA(if_train, DEVICE, model_name, TARGET_PATH, num_classes, num_features, max_dataset, target_test, target_model, "hdbscan", "max")
    roc_auc_noise = test_MIA(if_train, DEVICE, model_name, TARGET_PATH, num_classes, num_features, noise_dataset, target_test, target_model, "hdbscan", "noise")
    roc_auc_random = test_MIA(if_train, DEVICE, model_name, TARGET_PATH, num_classes, num_features, random_dataset, target_test, target_model, "hdbscan", "random")

    print("========MIN_dataset========")
    print(f"roc_auc:{roc_auc_min}")
    print("========MAX_dataset========")
    print(f"roc_auc:{roc_auc_max}")
    print("========NOISE_dataset========")
    print(f"roc_auc:{roc_auc_noise}")
    print("========RANDOM_dataset========")
    print(f"roc_auc:{roc_auc_random}")


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





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--dataset', type=str, default="student")
    parser.add_argument('--evaluate_type', type=int, default=0)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    print("==Start==")
    # 处理从命令行获得的参数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DEVICE = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset
    model_name = args.model
    TARGET_ROOT = "./dataloader/trained_model/"
    if not os.path.exists(TARGET_ROOT):
        print(f"Create directory named {TARGET_ROOT}")
        os.makedirs(TARGET_ROOT)
    TARGET_PATH = TARGET_ROOT + dataset_name + model_name

    # ----- 进行隐私风险评估 ----- #
    # 进行MIA评估
    if args.evaluate_type == 0:
        # 获得目标数据集、影子数据集、目标模型、影子模型
        num_classes, num_features, target_train, target_test, target_model = prepare_dataset(args.evaluate_type, dataset_name, model_name)
        test_MIA(args.train, DEVICE, model_name, TARGET_PATH, num_classes, num_features, target_train, target_test, target_model, evaluate_type="mia", data_type="")
    # 进行HDBSCAN聚类研究
    elif args.evaluate_type == 1:
        # 获得目标数据集、影子数据集、目标模型、影子模型
        num_classes, num_features, target_train, target_test, target_model = prepare_dataset(args.evaluate_type, dataset_name, model_name)
        test_MIA(args.train, DEVICE, model_name, TARGET_PATH, num_classes, num_features, target_train, target_test, target_model, evaluate_type="hdbscan", data_type="")
        test_HDBSCAN(args, dataset_name, model_name, target_train, target_test, target_model, DEVICE)
    # 进行QID脆弱性研究
    elif args.evaluate_type == 2:
        test_QID(dataset_name)

    print("==Finish==")





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