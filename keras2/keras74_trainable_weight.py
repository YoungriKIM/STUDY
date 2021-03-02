import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# 웨이트를 찾아서 보고싶어
# print(model.weights)                # 써머리에서 보면 trainabel_params만 34이다. 
# print(model.trainable_weights)        # 그래서 이렇게 프린트하면 위와 똑같이 나온다.

print(len(model.weights))               # 가중치가 들어있는 레이어+가중치에 붙어있는 bias > 즉 가중치가 있는 레이어*2
print(len(model.trainable_weights))