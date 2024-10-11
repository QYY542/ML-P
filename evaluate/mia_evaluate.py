import os
import glob
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class attack_for_blackbox():
    def __init__(self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model,
                 shadow_model, attack_model, device, model_name, num_features):
        self.device = device

        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS

        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

        self.model_name = model_name
        self.num_features = num_features

    # 获取模型的预测标签和预测向量
    def _get_data(self, model, inputs, targets):
        if self.model_name == "ResNet":
            inputs = inputs.reshape(inputs.shape[0], 1, self.num_features)
        result = model(inputs)

        output, _ = torch.sort(result, descending=True)
        # results = F.softmax(results[:,:5], dim=1)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        return output, prediction.unsqueeze(-1)

    # 根据影子模型和目标模型获取数据
    def prepare_dataset(self):
        # self.ATTACK_SETS = attack_sets
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            # targets 来自 dataloader
            # members 来自 get_attack_dataset_with_shadow
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.shadow_model, inputs, targets)
                # output = output.cpu().detach().numpy()

                pickle.dump((output, prediction, members), f)

        print("Finished Saving Train Dataset")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs, targets)
                # output = output.cpu().detach().numpy()

                pickle.dump((output, prediction, members), f)

        print("Finished Saving Test Dataset")

    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while (True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(
                        self.device)

                    results = self.attack_model(output, prediction)
                    results = F.softmax(results, dim=1)

                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)

            print("Saved Attack Train Ground Truth and Predict Sets")
            print("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1. * correct / total)
        print('Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (
            100. * correct / total, correct, total, 1. * train_loss / batch_idx))

        return final_result

    def test(self, epoch, result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while (True):
                    try:
                        output, prediction, members = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(
                            self.device)

                        results = self.attack_model(output, prediction)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        results = F.softmax(results, dim=1)

                        if epoch:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if epoch:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            print("Saved Attack Test Ground Truth and Predict Sets")
            print("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

        final_result.append(1. * correct / total)
        print('Test Acc: %.3f%% (%d/%d)' % (100. * correct / (1.0 * total), correct, total))
        test_acc = correct / (1.0 * total)
        return test_acc

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS + "train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS + "test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)


# black shadow
def attack_mode0(TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model,
                 shadow_model, attack_model, get_attack_set, model_name, num_features):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack0.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack0.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode0_"

    attack = attack_for_blackbox(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader,
                                 target_model, shadow_model, attack_model, device, model_name, num_features)

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()


    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i + 1))
        res_train = attack.train(flag, RESULT_PATH)
        test_acc = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return test_acc


# black partial
def attack_mode1(TARGET_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, target_model, attack_model,
                 get_attack_set, model_name, num_features):
    MODELS_PATH = ATTACK_PATH + "_meminf_attack1.pth"
    RESULT_PATH = ATTACK_PATH + "_meminf_attack1.p"
    ATTACK_SETS = ATTACK_PATH + "_meminf_attack_mode1_"

    attack = attack_for_blackbox(TARGET_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader,
                                 target_model, target_model, attack_model, device, model_name, num_features)

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(50):
        flag = 1 if i == 49 else 0
        print("Epoch %d :" % (i + 1))
        res_train = attack.train(flag, RESULT_PATH)
        test_acc = attack.test(flag, RESULT_PATH)

    attack.saveModel(MODELS_PATH)
    print("Saved Attack Model")

    return test_acc
