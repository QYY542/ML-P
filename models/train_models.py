import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset


class target_model_training():
    def __init__(self, trainloader, testloader, model, device, num_features, model_name):
        self.device = device
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_features = num_features
        self.model_name = model_name

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)

    # Training
    def train(self):
        self.net.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]

            if str(self.criterion) != "CrossEntropyLoss()":
                targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 对ResNet特殊处理
            if self.model_name == "ResNet":
                inputs = inputs.reshape(inputs.shape[0], 1, self.num_features)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if str(self.criterion) != "CrossEntropyLoss()":
                _, targets = targets.max(1)

            correct += predicted.eq(targets).sum().item()

        self.scheduler.step()

        print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (
            100. * correct / total, correct, total, 1. * train_loss / batch_idx))

        return 1. * correct / total

    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]
                if str(self.criterion) != "CrossEntropyLoss()":
                    targets = torch.from_numpy(np.eye(self.num_classes)[targets]).float()

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # 对ResNet特殊处理
                if self.model_name == "ResNet":
                    inputs = inputs.reshape(inputs.shape[0], 1, self.num_features)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if str(self.criterion) != "CrossEntropyLoss()":
                    _, targets = targets.max(1)

                correct += predicted.eq(targets).sum().item()

            print('Test Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

        return 1. * correct / total


def train_target_model(PATH, device, train_set, test_set, model, model_name, num_features):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=True, num_workers=2)

    model = target_model_training(train_loader, test_loader, model, device, num_features, model_name)
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


class shadow_model_training():
    def __init__(self, trainloader, testloader, model, device, loss, optimizer, num_features, model_name):
        self.device = device
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_features = num_features
        self.model_name = model_name

        self.criterion = loss
        self.optimizer = optimizer

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)

    # Training
    def train(self):
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 对ResNet特殊处理
            if self.model_name == "ResNet":
                inputs = inputs.reshape(inputs.shape[0], 1, self.num_features)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        self.scheduler.step()

        print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (
            100. * correct / total, correct, total, 1. * train_loss / batch_idx))

        return 1. * correct / total

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # 对ResNet特殊处理
                if self.model_name == "ResNet":
                    inputs = inputs.reshape(inputs.shape[0], 1, self.num_features)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Test Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))

        return 1. * correct / total


def train_shadow_model(PATH, device, shadow_train, shadow_test, shadow_model, model_name, num_features):
    train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        shadow_test, batch_size=64, shuffle=True, num_workers=2)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

    model = shadow_model_training(train_loader, test_loader, shadow_model, device, loss, optimizer, num_features,
                                  model_name)
    acc_train = 0
    acc_test = 0

    for i in range(100):
        print("<======================= Epoch " + str(i + 1) + " =======================>")
        print("shadow training")

        acc_train = model.train()
        print("shadow testing")
        acc_test = model.test()

        overfitting = round(acc_train - acc_test, 6)

        print('The overfitting rate is %s' % overfitting)

    FILE_PATH = PATH + "_shadow.pth"
    model.saveModel(FILE_PATH)
    print("saved shadow model!!!")
    print("Finished training!!!")

    return acc_train, acc_test, overfitting


