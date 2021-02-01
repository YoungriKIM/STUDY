# pca를 이용해 차원을 축소해보자. / RF로 모델도 만들어서 비교해보자~

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)    #(442, 10) (442,)

#-----------------------------------------------------------------------------------
# # n_components = n 으로 압축할 열 개수를 지정할 수 있다.
# pca = PCA(n_components = 7)
# x2 = pca.fit_transform(x)
# print(x2.shape)             # (442, 7)

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
# cumsum:  [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]
# cumsum >= 0.95 [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True]
# d:  1
# 하나로 압축해도 될 듯 ㅎ

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()