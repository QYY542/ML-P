import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST
from tqdm import tqdm

def train(savedir, train_ds, test_ds, model, DEVICE, model_name, num_features, evaluate_type):
    batch_size = 128
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 直接使用传入的目标模型
    model = model.to(DEVICE)

    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50)

    # 训练过程
    for i in range(300):
        model.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if model_name == "ResNet":
                x = x.reshape(x.shape[0], 1, num_features)

            loss = F.cross_entropy(model(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}, epoch: {i+1}")
            optim.zero_grad()
            loss.backward()
            optim.step()
            
        sched.step()

    # 保存模型
    accuracy = get_acc(model, test_dl, DEVICE, model_name, num_features)
    print(f"[test] acc_test: {accuracy:.4f}")

    savedir = os.path.join(savedir, evaluate_type)
    os.makedirs(savedir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(savedir, "model.pt"))

    return accuracy


@torch.no_grad()
def get_acc(model, dl, DEVICE, model_name, num_features):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if model_name == "ResNet":
            x = x.reshape(x.shape[0], 1, num_features)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()
