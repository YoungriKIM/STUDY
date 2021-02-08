# (두둥) 드디어 딥러닝모델과 머신러닝을 엮어보자

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
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model
model2 = build_model()

# 딥러닝 모델을 머신러닝 모델형태로 싸주자!
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_fn=build_model, verbose=1)
# 모델을 그냥 집어 넣으면 안되고 이렇게 싸주어야 랜덤서치나 그리드서치가 인식할 수 있다.
# 케라스에 여러 변수를 지정해서 머신러닝으로 돌리고 싶기 때문에 이렇게 합치는 것!

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropouts = [0.1, 0.2, 0.3]
    return {'batch_size' : batches, 'optimizer' : optimizers, 'drop' : dropouts}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
# 랜덤서치는 디폴트가 10 거기에 cv는 3 해서 10*3 = 30번 돌아갈 것!
search.fit(x_train, y_train, verbose=1)

# 에러 발생
# raise TypeError
# TypeError: If no scoring is specified, the estimator passed should have a 'score' method.
# The estimator <tensorflow.python.keras.engine.functional.Functional object at 0x00000182B1072EE0> does not.
# 형태가 안 맞아서 안 된다! 딥러닝 모델을 머신러닝 모델로 wrap 해야 한다! 36번 줄 확인~

# ----------------------------------------------------------------
print('best_params_: ', search.best_params_) # 이 모델에서 내가 설정한(36번~40번 3개)에서 최고만 알려주고
# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# 내가 더 튜닝할 수 있는건.. 엑티베이션, 러닝레이트..레이어..등
# ----------------------------------------------------------------
# print('best_estimator_: ', search.best_estimator_) # 얘는 설정할 수 있는 모든 파라미터에서 최고를 보여준다. 그런데 머신러닝이 딥러닝 모델의 파라미터를 전부 인식할 수 없을걸?
# best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E602CD9BB0>
# ----------------------------------------------------------------
# print('best_score_: ', search.best_score_)
#이건 뭐게?: 가장 잘나온 스코어 반환
# best_score_:  0.9590666691462199
# ----------------------------------------------------------------

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)
