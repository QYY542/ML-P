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
    def __init__(self, input_size, num_classes, dropout = 0.4):
        super(Net_1, self).__init__()
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


# 自定义ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)

        # Adapting the first layer to match num_features
        self.adapt_first_layer = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.Unflatten(1, (1, 8, 8))  # Example reshape, adjust based on your input reshaping strategy
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.adapt_first_layer(x)  # Adapt the input
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


# 前馈神经网络
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_rate=0.5):
        super(MLP, self).__init__()

        # 初始化模块列表
        self.layers = nn.ModuleList()

        # 添加第一层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Dropout(dropout_rate))

        # 根据hidden_sizes添加更多层
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.Dropout(dropout_rate))

        # 输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # 权重初始化
        self.apply(self.init_weights)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# CNN网络
class CNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNN, self).__init__()
        self.num_features_sqrt = int(num_features ** 0.5)

        # 假设num_features可以开方，以便能将一维数据重塑成二维形式
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        # 通过池化层后的大小计算
        size_after_conv = self.num_features_sqrt // 2 // 2  # 两次池化
        self.fc_input_size = size_after_conv * size_after_conv * 64

        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.num_features_sqrt, self.num_features_sqrt)  # 调整形状以匹配卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, self.fc_input_size)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
