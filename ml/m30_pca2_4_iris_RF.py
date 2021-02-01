# pca를 이용해 차원을 축소해보자. / RF로 모델도 만들어서 비교해보자~

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine, load_iris
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)     #(150, 4) (150,)


#-----------------------------------------------------------------------------------
# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum: ', cumsum)

# d = np.argmax(cumsum >= 0.95)+1
# print('cumsum >= 0.95', cumsum >=0.95) 
# print('d: ', d)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

#-----------------------------------------------------------------------------------
# pca 적용
pca = PCA(n_components = 3)
x2 = pca.fit_transform(x)
print(x2.shape)             # (150, 3)

pca_EVR = pca.explained_variance_ratio_ # 변화율
print('pca_EVR: ', pca_EVR)

print('cumsum: ', sum(pca_EVR))

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

# 모델 구성
model = RandomForestClassifier()

# 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train)

# 평가, 예측
score = model.score(x_test, y_test)
print('score: ', score)

# =========================================================
# m24
# score_1:  0.9333333333333333
# score_2:  0.9333333333333333

# =========================================================
# m30_pca 3로 압축 RandomForest
# pca_EVR:  [0.92461872 0.05306648 0.01710261]
# cumsum:  0.9947878161267247
# score:  0.9666666666666667