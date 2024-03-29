# 실습
# cifar10으로 vgg16 넣어서 만들 것! 결과치 비교
# 깃허브 탐방!!!!!

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import cifar10

# 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 전처리
y_train = y_train.reshape(y_train.shape[0],)
y_test = y_test.reshape(y_test.shape[0],)

from sklearn.preprocessing import OneHotEncoder
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

x_train = x_train.astype('float32')/255.  # 전처리
x_test = x_test.astype('float32')/255.  # 전처리

# 모델 구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
vgg16.trainable =False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))
model.summary()

# 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='acc', patience=8, mode='max')

model.fit(x_train, y_train, epochs=40, batch_size=64, validation_split=0.2, verbose=1, callbacks=[stop])

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss, acc:' ,loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

# ========================================================
# 43-2 cifar CNN
# loss: [0.040339864790439606, 0.7218999862670898]
# y_pred:  [5 8 0 0 6 6 1 6 3 1]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 43-3 cifar DNN
# loss: [0.07754093408584595, 0.3504999876022339]
# y_pred:  [5 8 8 0 4 6 5 6 5 9]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 42-4 cifar LSTM
# loss: [0.06874780356884003, 0.4440999925136566]
# y_pred:  [2 8 8 9 4 6 9 6 2 9]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 76 vgg16
# false
# loss, acc: [1.1486423015594482, 0.6087999939918518]
# y_pred:  [3 8 8 9 6 6 3 6 4 3]
# y_test:  [3 8 8 0 6 6 1 6 3 1]
# True
# loss, acc: [1.1306649446487427, 0.8032000064849854]
# y_pred:  [3 8 8 0 6 6 9 4 3 1]
# y_test:  [3 8 8 0 6 6 1 6 3 1]