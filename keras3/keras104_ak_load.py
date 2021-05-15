# 103에서 저장한 h5 파일을 살펴보자

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train[:6000].reshape(6000, 28, 28, 1).astype('float32')/255.
x_test = x_test[:1000].reshape(1000, 28, 28, 1).astype('float32')/255.
y_train = y_train[:6000]
y_test = y_test[:1000]
# 너무 오래 걸려서 데이터 줄임 ^^^

# 원핫인코딩 안 해도 되네..?

# 그래서 원핫인코딩 해부자
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 그럼 안 돌아가야 하는데? 근데 돌아가네? >> 사이킷런화 해서 여러 모델을 쓸 수 있다. imageclassifier에서는 원핫해도 되고 안 해도 된다

# 저장한 모델 불러와서 써부자
from tensorflow.keras.models import load_model
model = load_model('C:/data/h5/autokeras/keras103.h5')
model.summary()

# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 28, 28, 1)]       0              # 이미지의 인풋이 알아서 잡힘
# _________________________________________________________________
# cast_to_float32 (CastToFloat (None, 28, 28, 1)         0
# _________________________________________________________________
# normalization (Normalization (None, 28, 28, 1)         3
# _________________________________________________________________
# conv2d (Conv2D)              (None, 26, 26, 32)        320
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
# _________________________________________________________________
# dropout (Dropout)            (None, 12, 12, 64)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 9216)              0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 9216)              0
# _________________________________________________________________
# dense (Dense)                (None, 10)                92170
# _________________________________________________________________
# classification_head_1 (Softm (None, 10)                0             # 알아서 아웃풋 10으로
# =================================================================
# Total params: 110,989
# Trainable params: 110,986
# Non-trainable params: 3
# _________________________________________________________________


# 인풋 쉐이프가 없네 ?

# model = ak.ImageClassifier(
#     overwrite=True,
#     max_trials=1,         # epoch을 *n번 시도했다.
#     loss = 'mse',         # 로스 지정해주자
#     metrics=['acc']       # 매트릭스도!
# )

# 써머리 확인해보자
# model.summary()
# > 써머리가 안 먹힌다. : 오토케라스는 모델을 만들겠다는 계획이고 아직 만들어진 건 아님

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# stop = EarlyStopping(monitor='val_loss', mode='min', patience=4)
# lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
# filepath = 'C:/data/h5/autokeras/'
# mc = ModelCheckpoint(filepath, save_best_only=True, verbose=1)

# model.fit(x_train, y_train, epochs=1, validation_split=0.2, callbacks=[stop, lr, mc])
# # validation_split 디폴트가 0.2이다.

# results = model.evaluate(x_test, y_test)

# print(results)

# 그럼 여기서 써머리 확인해보자
# model.summary()
# 여기서도 안 되잖아요~!!

# 그럼 모델.save를 해주마
# AttributeError: 'ImageClassifier' object has no attribute 'save' >>> !?!?!??!?!??

# 그래서 imageclassifier를 모델 형태로 바꾸는 녀석을 쓰자
# model2 = model.export_model()
# model2.save('C:/data/h5/autokeras/keras103.h5')
# 저장됨!!!! :]

# 자동으로 체크 포인트까지 생성이 된다.
# 내가 지정한 경로에 mc 저장된 것 확인!

print('==== done ====')

# =========================================
# [0.1333298236131668, 0.9580000042915344]