# .coef_(기울기) 와.intercept_(편향) 에 대해 알아보자

#  리스트 형태의 데이터를 준비한다.
x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]

print(x,'\n',y)

import matplotlib.pyplot as plt
plt.plot(x, y)
# plt.show()
# weigt =1, bias=1 인 그래프

# ------------------------------------------------------

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})   # 키, 밸류의 딕셔너리로 지정
print(df)
#     X   Y
# 0  -3  -2
# 1  31  32
# 2 -11 -10
# 3   4   5
# 4   0   1
# 5  22  23
# 6  -2  -1
# 7  -5  -4
# 8 -25 -24
# 9 -14 -13
print(df.shape)
# (10, 2)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']

print(x_train.shape, y_train.shape)  #(10,) (10,)
# 스칼라 >  벡터  > 매트릭스(행렬) >   텐서
# (10,) > (10,1) >    (10,3)     > (10,3,2)

print('shape: ', x_train.shape, '\ntype: ',type(x_train))
# shape:  (10,)
# type:  <class 'pandas.core.series.Series'>

x_train = x_train.values.reshape(len(x_train), 1)
print('shape: ', x_train.shape, '\ntype: ',type(x_train))
# shape:  (10, 1)
# type:  <class 'numpy.ndarray'>

# ------------------------------------------------------

# 가장 기본적인 리니어 모델을 가져와서 써보자
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print('score: ' ,score)

print('기울기(weight): ', model.coef_)
print('편향, 졀편(bias): ',model.intercept_)
# 기울기(weight):  [1.]
# 편향, 졀편(bias):  1.0

# >> 데이터 별로 돌려보면 이해가 쉽다
# x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
# y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]
# 기울기(weight):  [1.]
# 편향, 졀편(bias):  1.0

# x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
# y = [-5, 63, -21, 9, 1, 45, -3, -9, -49, -27]
# 기울기(weight):  [2.]
# 편향, 졀편(bias):  1.0

# x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
# y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]
# 기울기(weight):  [2.]
# 편향, 졀편(bias):  3.0