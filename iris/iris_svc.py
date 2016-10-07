import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
features = iris.data
label = iris.target

pca = PCA(n_components=2)
features = pca.fit_transform(features)
setosa = features[label==0]
versicolor = features[label==1]
virginica = features[label==2]

plt.scatter(setosa[:, 0], setosa[:, 1], color='red')
plt.scatter(versicolor[:, 0], versicolor[:, 1], color='blue')
plt.scatter(virginica[:, 0], virginica[:, 1], color='green')

clf = svm.SVC(C=0.1)
clf.fit(features,label)
features_x_min = features[:, 0].min() - 2
features_x_max = features[:, 0].max() + 2
features_y_min = features[:, 1].min() - 2
features_y_max = features[:, 1].max() + 2
grid_interval = 0.02
xx, yy = np.meshgrid(
    np.arange(features_x_min, features_x_max, grid_interval),
    np.arange(features_y_min, features_y_max, grid_interval),
)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# 分類結果を表示する
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.bone, alpha=0.2)

# グラフを表示する
plt.autoscale()
plt.grid()
#plt.show()
plt.savefig("C0_1.pdf")
