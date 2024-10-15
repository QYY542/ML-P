import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from models.define_models import *

@torch.no_grad()
def inference(savedir, train_ds, test_ds, device, model_name, num_features, num_classes, evaluate_type, data_type):
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # 根据 model_name 创建正确的模型实例
    if model_name == "MLP":
        print("MLP")
        model = MLP(num_features, num_classes)
    elif model_name == "ResNet":
        print("ResNet")
        model = ResNetModel(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 构建模型路径
    savedir = os.path.join(savedir, evaluate_type)
    model_path = os.path.join(savedir, "model.pt")
    savedir = os.path.join(savedir, data_type)
    
    if os.path.exists(model_path):
        # 加载模型权重
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        # 生成并保存train_dl的logits
        train_logits = run_inference(model, train_dl, model_name, device, num_features)
        os.makedirs(savedir, exist_ok=True)
        train_logits_path = os.path.join(savedir, "train_logits.npy")
        np.save(train_logits_path, train_logits)
        print(train_logits[0])
        print(train_logits[1])
        print(f"Train logits saved to {train_logits_path}")

        # 生成并保存test_dl的logits
        test_logits = run_inference(model, test_dl, model_name, device, num_features)
        os.makedirs(savedir, exist_ok=True)
        test_logits_path = os.path.join(savedir, "test_logits.npy")
        np.save(test_logits_path, test_logits)
        print(test_logits[0])
        print(test_logits[1])
        print(f"Test logits saved to {test_logits_path}")
    else:
        print(f"Model not found in {model_path}")

def run_inference(model, data_loader, model_name, device, num_features):
    """
    运行推理过程并返回logits。
    """
    logits_n = []
    
    for _ in range(1):
        logits = []
        # 遍历数据集进行推理
        pbar = tqdm(data_loader)
        for itr, (x, y) in enumerate(pbar):
            x = x.to(device)
            if model_name == "ResNet":
                x = x.reshape(x.shape[0], 1, num_features)  # 对ResNet的输入进行调整
            outputs = model(x)
            logits.append(outputs.cpu().numpy())
        
        logits_n.append(np.concatenate(logits))

    # 将logits堆叠
    logits_n = np.stack(logits_n, axis=1)
    return logits_n

