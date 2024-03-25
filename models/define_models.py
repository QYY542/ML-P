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

# 自定义神经网络模型
class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)  # 第一层
        self.fc2 = nn.Linear(128, 64)            # 第二层
        self.fc3 = nn.Linear(64, 32)             # 第三层
        self.fc4 = nn.Linear(32, 2)              # 输出层，单个神经元

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 使用Tanh激活函数
        x = torch.tanh(self.fc2(x))  # 同上
        x = torch.tanh(self.fc3(x))  # 同上
        x = self.fc4(x)  # 输出层使用Sigmoid激活函数
        return x


class TexasClassifier(nn.Module):
	def __init__(self, num_classes=100):
		super(TexasClassifier, self).__init__()

		self.features = nn.Sequential(
			nn.Linear(6169, 1024),
			nn.Tanh(),
			nn.Linear(1024, 512),
			nn.Tanh(),
			nn.Linear(512, 256),
			nn.Tanh(),
			nn.Linear(256, 128),
			nn.Tanh(),
		)
		self.classifier = nn.Linear(128, num_classes)

	def forward(self, x):
		hidden_out = self.features(x)
		return self.classifier(hidden_out)


class FourLayerMultiClassNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FourLayerMultiClassNN, self).__init__()
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