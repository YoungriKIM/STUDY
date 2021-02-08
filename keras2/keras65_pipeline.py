# 이전 것 카피하여 파이프라인으로 구성


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터/ 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
modelpath = '../data/modelcheckpoint/k65_pipe_line_{epoch:02d}-{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath=modelpath, save_best_only=True, verbose=1)

model2 = KerasClassifier(build_fn=build_model, verbose=1, batch_size=32, epochs=1, callbacks=[stop, reduce_lr, mc])

# 파이프라인 정의
pipeline = Pipeline([('scaler', MinMaxScaler()), ('clf', model2)])

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.1, 0.2, 0.3]
    return {'clf__batch_size' : batches, 'clf__optimizer' : optimizers, 'clf__drop' : dropouts}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(pipeline, hyperparameters, cv = 3)
# 랜덤서치는 디폴트가 10 거기에 cv는 3 해서 10*3 = 30번 돌아갈 것!

# ----------------------------------------------------------------
# 훈련
search.fit(x_train, y_train)

# ----------------------------------------------------------------
print('best_params_: ', search.best_params_)
# best_params_:  {'clf__optimizer': 'adam', 'clf__drop': 0.2, 'clf__batch_size': 30}
# ----------------------------------------------------------------
acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)
# 최종 스코어:  0.964900016784668