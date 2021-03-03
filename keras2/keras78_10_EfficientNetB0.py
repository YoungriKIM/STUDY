# 실습
# cifar10으로 vgg16 넣어서 만들 것! 결과치 비교
# 깃허브 탐방!!!!!

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Input
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
input_tensor = Input(shape=(32, 32, 3))
apl = EfficientNetB0(weights='imagenet', include_top=False,input_tensor = input_tensor)
apl.trainable = True

model = Sequential()
model.add(apl)
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

# y_pred = model.predict(x_test[:10])
# print('y_pred: ', y_pred.argmax(axis=1))
# print('y_test: ', y_test[:10].argmax(axis=1))

# ========================================================
# 43-2 cifar CNN
# loss: [0.040339864790439606, 0.7218999862670898]
# y_pred:  [5 8 0 0 6 6 1 6 3 1]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 43-3 cifar DNN ----------------------------
# loss: [0.07754093408584595, 0.3504999876022339]
# y_pred:  [5 8 8 0 4 6 5 6 5 9]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 42-4 cifar LSTM ----------------------------
# loss: [0.06874780356884003, 0.4440999925136566]
# y_pred:  [2 8 8 9 4 6 9 6 2 9]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 76 vgg16 ----------------------------
# false
# loss, acc: [1.1486423015594482, 0.6087999939918518]

# True
# loss, acc: [1.1306649446487427, 0.8032000064849854]
# y_pred:  [3 8 8 0 6 6 9 4 3 1]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 78_1 vgg19 ----------------------------
# True
# loss, acc: [1.0089478492736816, 0.7914999723434448]

# 78_02  Xception
# 최소한의 크기가 71*71 이어야 한다는 뜻
# ValueError: Input size must be at least 71x71; got `input_shape=(32, 32, 3)`
# >> upsampling 해서 [0],[1] 쉐잎에 *3 해서 들어갈거니 전이학습 선언 할 떄부터 (96,96,3)
# loss, acc: [0.43911752104759216, 0.9021999835968018]


# 78_03  ResNet50 ----------------------------
# True
# loss, acc: [1.1165250539779663, 0.7803999781608582]

# 78_04  ResNet101 ----------------------------
# True
# loss, acc: [1.2699329853057861, 0.7462999820709229]

# 78_05  InceptionV3 ----------------------------
# True
# ValueError: Input size must be at least 75x75; got `input_shape=(32, 32, 3)'
# >> upsampling 해서 [0],[1] 쉐잎에 *3 해서 들어갈거니 전이학습 선언 할 떄부터 (96,96,3)
# loss, acc: [0.6404574513435364, 0.8590999841690063]

# 78_06  InceptionResNetV2 ----------------------------
# True
# ValueError: Input size must be at least 75x75; got `input_shape=(32, 32, 3)`
# >> upsampling 해서 [0],[1] 쉐잎에 *3 해서 들어갈거니 전이학습 선언 할 떄부터 (96,96,3)
# loss, acc: [0.6921852827072144, 0.829800009727478]

# 78_07  DenseNet121 ----------------------------
# True
# loss, acc: [0.8062302470207214, 0.829800009727478]

# 78_08  MobileNetV2 ----------------------------
# True
# loss, acc: [1.0046191215515137, 0.7753999829292297]

# 78_09  NASNetMobile ----------------------------
# True
# input_tensor 로 성정
# loss, acc: [3992.26513671875, 0.45899999141693115]

# 78_010  EfficientNetB0 ----------------------------
# True
# loss, acc: [6.4494242668151855, 0.1014999970793724]