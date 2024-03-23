import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.models as models

from MIA_Evaluate.meminf import *
from dataloader.train import *
from models.define_models import *
from dataloader.dataloader import *


def train_model(PATH, device, train_set, test_set, model):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=True, num_workers=2)

    model = model_training(train_loader, test_loader, model, device)
    acc_train = 0
    acc_test = 0

    for i in range(100):
        print("<======================= Epoch " + str(i + 1) + " =======================>")
        print("target training")

        acc_train = model.train()
        print("target testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

    FILE_PATH = PATH + "_target.pth"
    model.saveModel(FILE_PATH)
    print("Saved target model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting


def test_meminf(PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test, target_model,
                shadow_model, train_shadow, mode):
    batch_size = 64
    if train_shadow:
        shadow_trainloader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=2)
        shadow_testloader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=True, num_workers=2)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        train_shadow_model(PATH, device, shadow_model, shadow_trainloader, shadow_testloader, loss, optimizer)

    if mode == 0 or mode == 3:
        attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
            target_train, target_test, shadow_train, shadow_test, batch_size)
    else:
        attack_trainloader, attack_testloader = get_attack_dataset_without_shadow(target_train, target_test, batch_size)

    # for white box
    if mode == 2 or mode == 3:
        gradient_size = get_gradient_size(target_model)
        total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

    if mode == 0:
        attack_model = ShadowAttackModel(num_classes)
        attack_mode0(PATH + "_target.pth", PATH + "_shadow.pth", PATH, device, attack_trainloader, attack_testloader,
                     target_model, shadow_model, attack_model, 1, num_classes)

def str_to_bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--root', type=str, default="../data")
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--dataset', type=str, default="UTKFace")
    parser.add_argument('--attack_type', type=int, default=0)
    parser.add_argument('--train_target', action='store_true')
    parser.add_argument('--train_shadow', action='store_true')
    parser.add_argument('--mode', type=int, default=0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")
    #device = torch.device("mps")  # osx arm64 gpu

    dataset_name = args.dataset
    attr = 'new'
    if args.mode == 2:
        if dataset_name.lower() == 'UTKface'.lower():
            attr = 'race_gender'.split("_")
        elif dataset_name.lower() == 'CelebA'.lower():
            attr = 'attr_attr'.split("_")
        else:
            sys.exit("we have not supported this attribute yet! --\'")
    else:
        if dataset_name.lower() == 'UTKface'.lower():
            attr = 'race'
        elif dataset_name.lower() == 'CelebA'.lower():
            attr = 'attr'

    root = args.root
    model_name = args.model
    mode = args.mode
    train_shadow = args.train_shadow
    TARGET_ROOT = "./dataloader/trained_model/"
    if not os.path.exists(TARGET_ROOT):
        print(f"Create directory named {TARGET_ROOT}")
        os.makedirs(TARGET_ROOT)
    TARGET_PATH = TARGET_ROOT + dataset_name

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model = prepare_dataset(
        dataset_name, attr, root, model_name)

    if args.train_target:
        train_model(TARGET_PATH, device, target_train, target_test, target_model)

    # membership inference
    if args.attack_type == 0:
        test_meminf(TARGET_PATH, device, num_classes, target_train, target_test, shadow_train, shadow_test,
                    target_model, shadow_model, train_shadow, mode)

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
