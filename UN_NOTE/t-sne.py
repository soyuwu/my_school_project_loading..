import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from mpl_toolkits.mplot3d import Axes3D
# 创建一个示例数据集，假设有4类数据

data = pd.read_csv('train.csv', encoding='gbk')
label = data.iloc[:, -1:]
dataset = data.iloc[:, :-1]


smote=SMOTETomek()
#dataset,label=smote.fit_resample(dataset,label)



tsne = TSNE(n_components=3, random_state=42)
data_tsne = tsne.fit_transform(dataset)
# 绘制三维图
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
# 绘制四类标签的散点
labels = np.array(label).ravel()
colors = ['r', 'g', 'b', 'y']
for l, c in zip(np.unique(labels), colors):
    ax.scatter(data_tsne[labels == l, 0], data_tsne[labels == l, 1], data_tsne[labels == l, 2], c=c, label=f"Label {l}")
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('t-SNE Visualization of Four Classes')


plt.legend()
plt.show()


tsne = TSNE(n_components=2, random_state=40)
data_tsne = tsne.fit_transform(dataset)
plt.figure(figsize=(16, 10))
# 绘制四类标签的散点
labels = np.array(label).ravel()
# 根据类别绘制不同颜色的散点
for l in np.unique(labels):
    plt.scatter(data_tsne[labels == l, 0], data_tsne[labels == l, 1], label=f"Label {l}")
plt.title('t-SNE Visualization of Four Classes')
plt.legend()
plt.show()

