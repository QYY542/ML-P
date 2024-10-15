import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = [label for _, label in data]  # 这里假设 data 中每个元素都是 (image, label) 形式

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # 当data[idx]是(image, label)形式时直接返回

def score(savedir, train_ds, test_ds, evaluate_type, data_type):
    # 创建DataLoader
    train_ds = CustomDataset(train_ds)
    test_ds = CustomDataset(test_ds)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    savedir = os.path.join(savedir, evaluate_type, data_type)

    # 计算并保存train和test的scores
    train_scores = compute_scores(train_dl, savedir, "train")
    test_scores = compute_scores(test_dl, savedir, "test")

    # 将train_scores作为正例，test_scores作为负例
    labels = np.concatenate([np.ones(len(train_scores)), np.zeros(len(test_scores))])
    scores = np.concatenate([train_scores, test_scores])

    # 计算并绘制ROC曲线
    roc_auc = plot_roc_auc(labels, scores, savedir)
    return roc_auc

def compute_scores(data_loader, savedir, file_name):
    """
    计算数据集的scores并保存到文件
    """
    # 检查logits文件是否存在
    logits_path = os.path.join(savedir, file_name+"_logits.npy")
    if not os.path.exists(logits_path):
        print(f"No logits file found in {logits_path}.")
        return

    # 读取logits数据，形状为[n_examples, n_augs, n_classes]
    opredictions = np.load(logits_path)

    # 数值稳定的softmax计算
    predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
    predictions = np.exp(predictions)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, data_loader.dataset.targets[:COUNT]]
    predictions[np.arange(COUNT), :, data_loader.dataset.targets[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    # 计算scores
    scores = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)

    # 保存scores
    score_path = os.path.join(savedir, file_name+"_scores.npy")
    np.save(score_path, scores)
    print(scores[0])
    print(scores[1])
    print(f"Scores saved to {score_path}")

    return scores

def plot_roc_auc(labels, scores, savedir):
    """
    Plot ROC curves and compute AUC using scores values.
    """
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 如果 AUC 小于 0.5，则翻转 FPR 和 TPR
    if roc_auc < 0.5:
        fpr = 1 - fpr
        tpr = 1 - tpr
        roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制随机分类线

    # 设置图像的x轴、y轴及标题
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 保存ROC曲线图
    roc_image_path = os.path.join(savedir, "roc_curve.png")
    plt.savefig(roc_image_path)
    plt.close()
    print(f"ROC curve saved to {roc_image_path}")

    return roc_auc

# 获取标签的函数
def get_labels(data_loader):
    return np.array(data_loader.dataset.targets)  # 假设Dataset中存储了标签在dataset.targets中
