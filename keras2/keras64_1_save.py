# 이전 것 카피하여 가중치 저장 할 것
# 1) model.save()
# 2) pickle

# 61_4 로 저장한 modelcheckpoint와 비교하기

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터/ 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


#2. 모델 구성
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28, ), name = 'input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model
model2 = build_model()

# 딥러닝 모델을 머신러닝 모델형태로 싸주자!
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_fn=build_model, verbose=1)
# 모델을 그냥 집어 넣으면 안되고 이렇게 싸주어야 랜덤서치나 그리드서치가 인식할 수 있다.

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.1, 0.2, 0.3]
    return {'batch_size' : batches, 'optimizer' : optimizers, 'drop' : dropouts}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv = 3)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
modelpath = '../data/modelcheckpoint/k64_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath, save_best_only=True, verbose=1)

search.fit(x_train, y_train, verbose=1, epochs=1, validation_split=0.2, callbacks=[stop, reduce_lr, mc])

# ---------------------------------------------------------------- save save save 
# save 해보기 / 1) model.save
search.best_estimator_.model.save('../data/h5/k64.h5')
print('=====save complete=====')

# ing

# ---------------------------------------------------------------- save save save 
# save 해보기 / 2) pickle
# import pickle

# pickle.dump(search, open('../data/xgb_save/k64_save.pickle.dat', 'wb'))
# print('=====save complete=====')

# TypeError: cannot pickle '_thread.RLock' object

# ---------------------------------------------------------------- save save save 
# save 해보기 / 3) joblib
# import joblib
# joblib.dump(search, '../data/xgb_save/k64_save.pickle.dat')
# print('=====save complete=====')

# TypeError: cannot pickle '_thread.RLock' object

# ----------------------------------------------------------------
# ----------------------------------------------------------------
print('best_params_: ', search.best_params_)
# best_params_:  {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 40}
# ----------------------------------------------------------------

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)
# 최종 스코어:  0.9588000178337097
# ----------------------------------------------------------------
