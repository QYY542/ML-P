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


# 自定义Net_1
class Net_1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net_1, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, 256)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.5)
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

# 自定义ResNet
class Bottlrneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=2, classes=6):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            #
            Bottlrneck(256, 128, 512, True),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            Bottlrneck(512, 128, 512, False),
            #
            Bottlrneck(512, 256, 1024, True),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            Bottlrneck(1024, 256, 1024, False),
            #
            Bottlrneck(1024, 512, 2048, True),
            Bottlrneck(2048, 512, 2048, False),
            Bottlrneck(2048, 512, 2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048, classes)
        )

    def forward(self, x):
        x = torch.Tensor.view(x, (-1, 2, 511))
        x = self.features(x)
        x = x.view(-1, 2048)
        x = self.classifer(x)
        return x

