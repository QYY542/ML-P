import torch
import torch.nn as nn
import torch.nn.functional as F


class ShadowAttackModel(nn.Module):
    def __init__(self, class_num):
        super(ShadowAttackModel, self).__init__()
        self.Output_Component = nn.Sequential(
            nn.Linear(class_num, 256),  # 增加神经元数量
            nn.BatchNorm1d(256),  # 添加批量归一化
            nn.ReLU(),
            nn.Dropout(p=0.3),  # 添加Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.Prediction_Component = nn.Sequential(
            nn.Linear(1, 256),  # 与Output_Component保持一致
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.Encoder_Component = nn.Sequential(
            nn.Linear(256, 512),  # 增加复杂度
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, output, prediction):
        Output_Component_result = self.Output_Component(output)
        Prediction_Component_result = self.Prediction_Component(prediction)

        final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), dim=1)
        final_result = self.Encoder_Component(final_inputs)

        return final_result


class PartialAttackModel(nn.Module):
    def __init__(self, class_num):
        super(PartialAttackModel, self).__init__()
        # 增加了模型的复杂度，并引入了批量归一化和Dropout来增强模型的泛化能力
        self.Output_Component = nn.Sequential(
            nn.Linear(class_num, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.Prediction_Component = nn.Sequential(
            nn.Linear(1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.Encoder_Component = nn.Sequential(
            nn.Linear(256, 512),  # 增加输入特征的维度来匹配上一层的输出
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, output, prediction):
        # 处理输出和预测的组件
        Output_Component_result = self.Output_Component(output)
        Prediction_Component_result = self.Prediction_Component(prediction)

        # 将输出和预测的结果连接起来作为Encoder组件的输入
        final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), dim=1)
        final_result = self.Encoder_Component(final_inputs)

        return final_result


# MLP
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.4):
        super(MLP, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, 256)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 通过网络层前向传播输入x
        x = self.bn1(self.fc1(x))
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.bn2(self.fc2(x))
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        x = self.bn3(self.fc3(x))
        x = self.leaky_relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


# ResNet网络
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, dropout_rate=0.5):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.dropout(Y)  # 添加Dropout
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def resnet_block(input_channels, num_channels, first_block=False, dropout_rate=0.5):
    blk = []
    if not first_block:
        blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2, dropout_rate=dropout_rate))
    else:
        blk.append(Residual(input_channels, num_channels, dropout_rate=dropout_rate))
    return nn.Sequential(*blk)


class ResNetModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(ResNetModel, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = resnet_block(64, 64, first_block=True, dropout_rate=dropout_rate)
        self.b3 = resnet_block(64, 128, dropout_rate=dropout_rate)
        self.b4 = resnet_block(128, 256, dropout_rate=dropout_rate)
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 增加Dropout
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        for block in [self.b1, self.b2, self.b3, self.b4]:
            x = block(x)
        x = self.final_layers(x)
        return x

