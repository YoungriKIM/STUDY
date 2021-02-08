# 61번 가져와서 dnn으로 boston~diabetes까지 5개 세트 모델 만들기
# 손코딩한 이전 파일과 성능 비교
# 랜덤만 쓰지말고 그리드서치도 적용한 파일도 만들어라

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.datasets import load_boston, load_breast_cancer, load_wine, load_iris, load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터/ 전처리
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle= True, random_state=311)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
def build_model(drop=0.2, optimizer='adam'):
    inputs = Input(shape=(x_train.shape[1]), name='input')
    x = Dense(256, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden2')(x)
    x = Dense(64, activation='relu', name='hidden3')(x)
    x = Dense(32, activation='relu', name='hidden4')(x)
    x = Dense(32, activation='relu', name='hidden5')(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model
model2 = build_model()

# wrap 적용
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_model, verbose=1)

# 하이퍼파라미터 지정
def create_hyperparameters():
    batches = [28, 34, 42, 24]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.1, 0.2, 0.3]
    return {'batch_size': batches, 'optimizer': optimizers, 'drop': dropouts}
hyper = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

search = RandomizedSearchCV(model2, hyper, cv = 3)

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
search.fit(x_train, y_train, epochs= 100, verbose=1, validation_split=0.2, callbacks=[stop])



# ----------------------------------------------------------------
print('best_params_: ', search.best_params_)
# best_params_:  {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 24}

# ----------------------------------------------------------------
score = search.score(x_test, y_test)
print('최종 스코어: ', score)
# 최종 스코어:  0.9666666388511658