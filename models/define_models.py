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
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
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
