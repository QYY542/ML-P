import argparse

from evaluate.mia_evaluate import *
from dataloader.dataloader_adult import  Adult
from dataloader.dataloader_attack import get_attack_dataset_with_shadow, get_attack_dataset_without_shadow
from dataloader.dataloader_hospital import  Hospital
from dataloader.dataloader_obesity import  Obesity
from dataloader.dataloader_student import  Student
from models.train_models import *
from models.define_models import *
from dataloader.dataloader import *


def test_meminf(PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model,
                shadow_model, mode):
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
        attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader,
                     target_model, shadow_model, attack_model, 1, num_classes)
    # 进行MIA评估 黑盒+Partial辅助数据集
    elif mode == 1:
        attack_model = PartialAttackModel(num_classes)
        attack_mode1(PATH + "_target.pth", PATH, device, attack_trainloader, attack_testloader, target_model,
                     attack_model, 1, num_classes)


def prepare_dataset(dataset_name,model_name):
    # 数据集
    if dataset_name == "ADULT":
        print("ADULT")
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
    each_length = length // 4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(
        dataset, [each_length, each_length, each_length, each_length, length - (each_length * 4)]
    )
    num_features = next(iter(dataset))[0].shape[0]

    # 模型
    if model_name == "Net_1":
        target_model = Net_1(num_features, num_classes)
        shadow_model = Net_1(num_features, num_classes)
    elif model_name == "Res1D":
        target_model = ResNet(num_features, num_classes)
        shadow_model = ResNet(num_features, num_classes)

    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataset', type=str, default="UTKFace")
    parser.add_argument('--attack_type', type=int, default=0)
    parser.add_argument('--train_target', action='store_true')
    parser.add_argument('--train_shadow', action='store_true')
    parser.add_argument('--mode', type=int, default=0)

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
    TARGET_PATH = TARGET_ROOT + dataset_name

    # 获得目标数据集、影子数据集、目标模型、影子模型
    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(dataset_name, model_name)

    # 训练目标模型
    if args.train_target:
        train_target_model(TARGET_PATH, device, target_train, target_test, target_model)

    # 训练影子模型
    if args.train_shadow:
        train_shadow_model(TARGET_PATH, device, shadow_train, shadow_test, shadow_model)

    # 进行MIA评估
    if args.attack_type == 0:
        test_meminf(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test,
                    target_model, shadow_model, mode)
    # # 进行QID脆弱性研究
    # elif args.attack_type == 1:





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
