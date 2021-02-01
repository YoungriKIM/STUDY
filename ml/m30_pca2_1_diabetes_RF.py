# pca를 이용해 차원을 축소해보자. / RF로 모델도 만들어서 비교해보자~
# 5개 세트 만들어라

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)     #(442, 10) (442,)

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
pca = PCA(n_components = 8)
x2 = pca.fit_transform(x)
print(x2.shape)             # (442, 8)

pca_EVR = pca.explained_variance_ratio_ # 변화율
print(pca_EVR)

print(sum(pca_EVR))

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

# 모델 구성
model = RandomForestRegressor()

# 컴파일ㄴ 훈련ㅇ
model.fit(x_train, y_train)

# 평가, 예측
score = model.score(x_test, y_test)
print('score: ', score)


# =========================================================
# m22, 23
# score_1:  0.4774730261936826
# score_2:  0.47110514856246877
# =========================================================
# m24
# score_1:  0.36261268384736134
# score_2:  0.428876611730748
# =========================================================
# m30_pca 8로 압축
# score:  0.4744528214860011