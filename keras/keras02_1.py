# 네이밍 룰을 지킬 것, 파이썬은 의미가 달라지는 부분에서 _ 을 삽입하고, 자바는 대문자로 써준다.
# 그래서 자바는 소문자 다음에 대문자가 나오는 모습이 낙타같아서 카멜케이스라고 한다.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#임포트는 내가 쓰기전에만 주면 되는데 보기 힘드니까 통상적으로 가장 위에 몰아서 한다.

#1. 데이터를 주자
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8]) #얘네는 평가용 데이터
y_test = np.array([6,7,8])

#2. 모델을 구성하자
model = Sequential()
#모델 구성을 할 때 지금 이 17번은 6번에 관한 내용인데 이럴때 6번을 from tensorflow.keras import models로 하고 17번을 model = models.Sequential() 로 써도 된다. 
#이렇게 해도 되지만 번거로우니까 6,7번 처럼 미리 위에서 카테고리를 지정을 해주는거임
model.add(Dense(5, input_dim = 1, activation='linear')) #계산을 할 때 w와 b(바이어스)가 연산이 되는데 저 activation로 계산이 된다는 정도만 알아둬라 relu,linear는 알고만 있어라
model.add(Dense(3, activation='relu')) #keras01_1의 파일과 다른 이유는, 위의 엑티베이션을 그대로 쓴다는 것이 아니라! 디폴트값으로 적용을 하겠다는 뜻이다. 그 디폴트는 linear 이다.
model.add(Dense(4))
model.add(Dense(1)) #마지막의 엑티베이션은 리니어야 하니까 수정하지 말 것 당분간

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #adam도 당분간은 몰라도 된다.
model.fit(x_train, y_train, epochs=100, batch_size=1) #배치 사이즈가 리스트 사이즈보다 커도 알아서 줄여져서 된다.

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1) #평가지표는 로스로 한다.
print('loss: ', loss)

result = model.predict([9])
print('result: ', result)

#결과값이 9의 근사치가 아니라면 내가 바꿀 수 있는 것들을 바꿔서(레이어의 깊이, 갯수, 엑티베이션, 배치사이즈,에포 등) 9의 근사치를 찾는 연습을 해라