import torch
import torch.nn as nn
import torch.nn.functional as F

class ShadowAttackModel(nn.Module):
	def __init__(self, class_num):
		super(ShadowAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(128, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction)
		
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
		final_result = self.Encoder_Component(final_inputs)

		return final_result

class PartialAttackModel(nn.Module):
	def __init__(self, class_num):
		super(PartialAttackModel, self).__init__()
		self.Output_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(class_num, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Prediction_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
		)

		self.Encoder_Component = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(128, 256),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			# nn.Dropout(p=0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)


	def forward(self, output, prediction):
		Output_Component_result = self.Output_Component(output)
		Prediction_Component_result = self.Prediction_Component(prediction)
		
		final_inputs = torch.cat((Output_Component_result, Prediction_Component_result), 1)
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
		self.fc1 = nn.Linear(input_size, 128)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		self.fc2 = nn.Linear(128, 256)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(256, 64)
		self.relu3 = nn.ReLU()
		self.dropout3 = nn.Dropout(0.5)
		self.fc4 = nn.Linear(64, num_classes)

	def forward(self, x):
		# 通过网络层前向传播输入x
		x = self.dropout1(self.relu1(self.fc1(x)))
		x = self.dropout2(self.relu2(self.fc2(x)))
		x = self.dropout3(self.relu3(self.fc3(x)))
		x = self.fc4(x)  # 最后一层不加激活函数，因为在交叉熵损失函数中会计算Softmax
		return x