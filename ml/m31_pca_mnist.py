# PCA 에 3차원이 먹일까? mnist로 확인
# 실습: pca를 0.95% 이상인 것은 몇개? 배운거 다 적용해서 확인

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

(x_train, _), (x_test, _) = mnist.load_data()
# _ 는 안하겠다는 뜻. y_train과 y_test는 하지 않는다.

x = np.append(x_train, x_test, axis=0)
print(x.shape)      #(70000, 28, 28) / x_train = 60000,28,28 / x_test = 10000,28,28

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
print(x.shape)      #(70000, 784)

#-----------------------------------------------------------------------------------
# # pca적용 전에 압축률 확인 
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

# 3차원을 pca에 넣을 때 > 2차원으로 리쉐잎해주자
# ValueError: Found array with dim 3. Estimator expected <= 2.

# 2차원으로 넣었을 때 0.95 이상의 수는 d:  154
#-----------------------------------------------------------------------------------
# pca 적용
pca = PCA(n_components = 154)
x2 = pca.fit_transform(x)
print(x2.shape)            # (70000, 154)

pca_EVR = pca.explained_variance_ratio_ # 변화율
print('pca_EVR: ', pca_EVR)

print('cumsum: ', sum(pca_EVR))    # cumsum: 0.9500035195029432

