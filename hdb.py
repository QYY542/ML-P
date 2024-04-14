import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.datasets import make_blobs

# 生成数据
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建HDBSCAN对象
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)

# 训练模型
clusterer.fit(data)

# 可视化
plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(set(clusterer.labels_))))  # 使用viridis颜色映射
plt.scatter(data[:, 0], data[:, 1], c=colors[clusterer.labels_], s=50, edgecolors='k')  # 添加边缘颜色为黑色，以更好地区分点
plt.title('HDBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 创建颜色条
scatter = plt.scatter(data[:, 0], data[:, 1], c=clusterer.labels_, cmap='viridis', s=50)
plt.colorbar(scatter)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()