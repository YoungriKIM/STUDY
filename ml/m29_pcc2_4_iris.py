# pca를 이용해 차원을 축소해보자 / 압축률도 확인해보자
# 고차원의 데이터를 저차원의 데이터로 환원시키는 기법. 400개 칼럼이면 200개로 압축하는 것!

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, load_iris
from sklearn.decomposition import PCA 
#decomposition: 분해 / PCA: 주성분분석(Principal Component Analysis) 
from sklearn.ensemble import RandomForestRegressor

dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)     #(150, 4) (150,)

#-----------------------------------------------------------------------------------
# # n_components = n 으로 압축할 열 개수를 지정할 수 있다.
# pca = PCA(n_components = 2)
# x2 = pca.fit_transform(x)
# print(x2.shape)             # (150, 2)

# pca_EVR = pca.explained_variance_ratio_ # 변화율
# print(pca_EVR)

# print(sum(pca_EVR)) # 다 더했을 때 압축률

#-----------------------------------------------------------------------------------

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum: ', cumsum)

d = np.argmax(cumsum >= 0.95)+1
print('cumsum >= 0.95', cumsum >=0.95)
print('d: ', d)

#========================================================
# cumsum:  [0.92461872 0.97768521 0.99478782 1.        ]
# cumsum >= 0.95 [False  True  True  True]
# d:  2

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()