# pca를 이용해 차원을 축소해보자 / 압축률도 확인해보자
# 고차원의 데이터를 저차원의 데이터로 환원시키는 기법. 400개 칼럼이면 200개로 압축하는 것!
# 5개 세트 만들어라

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA 
#decomposition: 분해 / PCA: 주성분분석(Principal Component Analysis) 
from sklearn.ensemble import RandomForestRegressor

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)     #(442, 10) (442,)

#----------------------------------------------------------------------------------
# # n_components = n 으로 압축할 열 개수를 지정할 수 있다.
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
# cumsum : 주어진 축을 따라 요소의 누적 합계를 반환
print('cumsum: ', cumsum)
# ================================================================
# 컬럼이 10개 일 때: 총 9번 더해지고 총합은 1.
# cumsum:  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]
#  압축률  -----------                      ------------                                                        ----------
#           1개로 압축                          4개로 압축                                                      압축 안해

d = np.argmax(cumsum >= 0.95)+1
# 다차원 배열에서 가장 큰 값의 인덱스들을 반환해주는 함수
print('cumsum >= 0.95', cumsum >=0.95)      # cumsumdl 0.95 이상인 애들을 True로 반환
print('d: ', d)                             # 몇개로 압축했을 땨 0.95 이상이 되는지
# ================================================================
# cumsum >= 0.95 [False False False False False False False  True  True  True]
# d:  8

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()