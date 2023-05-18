path = "../data/"

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


X = np.array([[12, 350, 1.825, 0.102, 315, 0, 2, 4], [25, 300, 5.57, 0.45, 220, 25, 3, 2.5],  [25, 300, 5.25, 1.1, 220, 20, 4, 3]])
Y = np.array([22, 300, 4.25, 1.86, 210, 18, 3, 2])
# n_components指定降维后的维数
pca = PCA(n_components=2)
print(pca)
# 应用于训练集数据进行PCA降维
pca.fit(X)
# 用X来训练PCA模型，同时返回降维后的数据
newX = pca.fit_transform(X)
print(newX)
# 将降维后的数据转换成原始数据，
pca_new = pca.transform(X)
print(pca_new.shape)
# 输出具有最大方差的成分
print(pca.components_)
# 输出所保留的n个成分各自的方差百分比
print(pca.explained_variance_ratio_)
# 输出所保留的n个成分各自的方差
print(pca.explained_variance_)
# 输出未处理的特征维数
print(pca.n_features_)
# 输出训练集的样本数量
print(pca.n_samples_)
# 输出协方差矩阵
print(pca.noise_variance_)
# # 每个特征的奇异值
# print(pca.ingular_values_)
# 用生成模型计算数据精度矩阵
print(pca.get_precision())

# 计算生成特征系数矩阵
covX = np.around(np.corrcoef(X.T), decimals=3)
# 输出特征系数矩阵
print(covX)
# 求解协方差矩阵的特征值和特征向量
featValue, featVec = np.linalg.eig(covX)
# 将特征进行降序排序
featValue = sorted(featValue)[::-1]

# 图像绘制
# 同样的数据绘制散点图和折线图
plt.scatter(range(1,  X.shape[1] + 1),  featValue)
plt.plot(range(1,  X.shape[1] + 1),  featValue)

# 显示图的标题
plt.title("Test Plot")
# xy轴的名字
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
# 显示网格
plt.grid()
# 显示图形
plt.show()

