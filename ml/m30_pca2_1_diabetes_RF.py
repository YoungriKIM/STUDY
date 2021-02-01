# pca를 이용해 차원을 축소해보자. / 모델도 만들어서 비교해보자~

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)     #(442, 10) (442,)

#-----------------------------------------------------------------------------------

# pca = PCA(n_components = 7)
# x2 = pca.fit_transform(x)
# print(x2.shape)             # (442, 7)

# pca_EVR = pca.explained_variance_ratio_ # 변화율
# print(pca_EVR)
# # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]

# print(sum(pca_EVR))
#0.9479436357350414 > 다 더했을 때 0.94 즉 압축률은 0.94%

#-----------------------------------------------------------------------------------

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum: ', cumsum)


d = np.argmax(cumsum >= 0.95)+1
print('cumsum >= 0.95', cumsum >=0.95) 
print('d: ', d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()