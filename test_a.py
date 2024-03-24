from sklearn.preprocessing import StandardScaler
import numpy as np

# 创建一个示例数据集
# 假设我们有三个特征
X = np.array([[1.0, -1.0, 2.0],
              [2.0, 0.0, 0.0],
              [0.0, 1.0, -1.0]])

# 初始化StandardScaler
scaler = StandardScaler()

# 使用fit_transform来计算数据的均值和标准差，并进行标准化
X_scaled = scaler.fit_transform(X)

# 打印原始数据
print("原始数据:")
print(X)

# 打印标准化后的数据
print("\n标准化后的数据:")
print(X_scaled)
