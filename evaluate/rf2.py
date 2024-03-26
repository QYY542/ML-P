import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class QIDVulnerabilityEvaluator:
    def __init__(self, T, ntree=100, mtry=None):
        """
        初始化评估器。
        :param T: 数据集，其中最后一列是标签。
        :param ntree: 随机森林中的树的数量。
        :param mtry: 分割节点时考虑的特征数量。
        """
        self.T = T
        self.ntree = ntree
        self.mtry = mtry
        self.model = RandomForestClassifier(n_estimators=ntree, max_features=mtry, random_state=42)

    def split_data(self):
        """
        划分数据集为训练集和测试集。
        """
        n = self.T.shape[0]
        self.train_index, self.test_index = train_test_split(np.arange(n), train_size=0.66, random_state=42)
        self.Ttrain = self.T[self.train_index, :]
        self.Ttest = self.T[self.test_index, :]

    def train_model(self):
        """
        训练随机森林模型。
        """
        self.model.fit(self.Ttrain[:, :-1], self.Ttrain[:, -1])

    def compute_tau(self, Q):
        """
        计算并返回所有QID的τ值。
        :param Q: 要评估的QID的索引列表。
        """
        τ = {}
        σref = accuracy_score(self.Ttest[:, -1], self.model.predict(self.Ttest[:, :-1]))

        for q in Q:
            σran_list = []
            for _ in range(500):  # 对每个QID随机化500次
                sData = self.Ttest.copy()
                np.random.shuffle(sData[:, q])  # 随机化QID列
                σran = accuracy_score(sData[:, -1], self.model.predict(sData[:, :-1]))
                σran_list.append(σran)
            σran_mean = np.mean(σran_list)
            D = (1 - σref) - (1 - σran_mean)
            mqi = np.abs(D)  # 简化版本，直接使用D的绝对值作为mqi
            sqi = np.std(σran_list, ddof=1)
            τqi = sqi / mqi if mqi != 0 else 0
            τ[q] = τqi
        return τ


if __name__ == "__main__":
    # 假设T是一个NumPy数组，其中最后一列是标签
    T = np.random.rand(1000, 5)  # 示例数据集，1000行，4个特征和1个标签列
    Q = [0, 1, 2, 3]  # 假设所有特征列都是QID
    evaluator = QIDVulnerabilityEvaluator(T, ntree=100, mtry="sqrt")
    evaluator.split_data()
    evaluator.train_model()
    tau_values = evaluator.compute_tau(Q)
    print("τ values for each QID:", tau_values)
