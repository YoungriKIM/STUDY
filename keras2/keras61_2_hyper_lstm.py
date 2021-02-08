# (두둥) 드디어 딥러닝모델과 머신러닝을 엮어보자
# > LSTM 모델로 다시 구성 + 파라미터도 변경 할 것(필수: 노드의 갯수)

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPooling2D, ReLU, LeakyReLU, PReLU, LSTM
from tensorflow.keras.datasets import mnist
from keras.optimizers import Adam, Adadelta, SGD, RMSprop


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터/ 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255.

#2. 모델 구성
def build_model(drop=0.5, optimizer = Adam, lr=00.1, node=512, activation='relu'):
    optimizer = optimizer(lr=lr)
    inputs = Input(shape=(28, 28), name = 'input')
    x = LSTM(node, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node*0.8, activation=activation, name='hidden2')(x)
    x = Dense(node*0.5, activation=activation, name='hidden3')(x)
    x = Dense(node*0.3, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model
model2 = build_model()

# 딥러닝 모델을 머신러닝 모델형태로 싸주자!
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_fn=build_model, verbose=1)
# 모델을 그냥 집어 넣으면 안되고 이렇게 싸주어야 랜덤서치나 그리드서치가 인식할 수 있다.

def create_hyperparameters():
    batches = [32, 64, 48, 24]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.1, 0.2, 0.3]
    nodes = [600, 400, 200]
    activation = ['relu', 'linear', 'tanh']
    lr = [0.1, 0.01, 0.001]
    optimizers = [Adam, Adadelta, SGD, RMSprop]
    return {'batch_size' : batches, 'optimizer' : optimizers, 'drop' : dropouts, 'node' : nodes, \
            'activation' : activation, 'lr' : lr, 'optimizer' : optimizers}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
# 랜덤서치는 디폴트가 10 거기에 cv는 3 해서 10*3 = 30번 돌아갈 것!

search.fit(x_train, y_train, verbose=1)

# ----------------------------------------------------------------
print('best_params_: ', search.best_params_)
# best_params_:  {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.gradient_descent.SGD'>, 'node': 400, 'lr': 0.1, 'drop': 0.2, 'batch_size': 24, 'activation': 'tanh'}
# ----------------------------------------------------------------

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)
# 최종 스코어:  0.9516000151634216
