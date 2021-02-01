# pca를 이용해 차원을 축소해보자.
# 고차원의 데이터를 저차원의 데이터로 환원시키는 기법. 400개 칼럼이면 200개로 압축하는 것!
# pca는 압축했을 때 압축률과 손실을 확인 / feature_importances 는 각 칼럼의 중요도로 둘은 다르니 알아서 써라~
# 하지만 pca는 특성자체를 건드린다는 것 주의
# https://excelsior-cjh.tistory.com/167

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
# n_components = n 으로 압축할 열 개수를 지정할 수 있다.
pca = PCA(n_components = 7)
x2 = pca.fit_transform(x)
print(x2.shape)             # (442, 7) > 칼럼(특성, 열)이 압축되었음을 확인. 성능은 어떨까?
# 어떻게 압축 된 건지 내역을 보자
pca_EVR = pca.explained_variance_ratio_ # 변화율
print(pca_EVR)
# [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
# 7개로 압축 된 열의 중요도가 나온다.
print(sum(pca_EVR))
#0.9479436357350414 > 다 더했을 때 0.94 즉 압축률은 0.94%
# 8로 압축했을 때     >     0.9913119559917797
# 9로 압축했을 때     >     0.9991439470098977
# 어떤 압출률이 좋은지는 모델을 돌려서 확인해봐야 한다.