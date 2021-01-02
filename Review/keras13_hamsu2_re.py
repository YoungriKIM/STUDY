from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

#데이터 주고
x = np.array([range(1,101), range(101,201), range(201,301), range(301,401), range(401,501)])
y = np.array([range(311,411), range(519,619)])
x_pred2 = np.array([101,201,301,401,501])

x = np.transpose(x)
y = np.transpose(y)
x_pred2 = x_pred2.reshape(1, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True)

#모델 구성
cherry = Input(shape=(5,))
banana = Dense(10, activation='relu')(cherry)
tomato = Dense(7)(banana)
grape = Dense(3)(tomato)
orange = Dense(2)(grape)
model = Model(inputs = cherry, outputs = orange)

#컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2, verbose=0)

#평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss: ', loss)
print('mae: ', mae)

y_pred2 = model.predict(x_pred2)
print('y_pred2: ', y_pred2)

### 이 아래 안 먹히는데 다시 해보기
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred2)
print('R2: ', R2)
