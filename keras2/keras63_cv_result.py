# 61 카피해서
# model.cv_result 붙여서 완성


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
# 형태가 안 맞아서 안 된다! 딥러닝 모델을 머신러닝 모델로 wrap 해야 한다! 31번 줄 확인~

# ----------------------------------------------------------------
print('best_params_: ', search.best_params_)
# 이 모델에서 내가 설정한(36번~40번 3개)에서 최고만 알려주고
# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# 내가 더 튜닝할 수 있는건.. 엑티베이션, 러닝레이트..레이어..등
# ----------------------------------------------------------------
# print('best_estimator_: ', search.best_estimator_)
# 얘는 설정할 수 있는 모든 파라미터에서 최고를 보여준다. 그런데 머신러닝이 딥러닝 모델의 파라미터를 전부 인식할 수 없을걸?
# best_estimator_:  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E602CD9BB0>
# ----------------------------------------------------------------
# print('best_score_: ', search.best_score_)
#이건 뭐게?
# best_score_:  0.9590666691462199
# ----------------------------------------------------------------

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)
# 최종 스코어:  0.9639000296592712

# ----------------------------------------------------------------
cv_result = search.cv_results_
print('cv_result: ', cv_result)

# RandomizedSearchCV
# cv_result:  {'mean_fit_time': array([1.89750783, 2.83707921, 5.7789797 , 3.30037475, 2.15925757,
#        5.74356016, 6.06448555, 1.84020782, 4.05504616, 1.75548657]), 'std_fit_time': array([0.09545298, 0.09081186, 0.08696685, 0.05253022, 0.07166385,
#        0.11023918, 0.14150297, 0.01716989, 0.11430661, 0.09540873]), 'mean_score_time': array([0.60190805, 0.88174423, 1.91636332, 1.08213401, 0.58976444,
#        1.942396  , 1.98222256, 0.51843389, 1.07487384, 0.51046189]), 'std_score_time': array([0.03081021, 0.03280832, 0.03515193, 0.03844378, 0.01983137,
#        0.05904438, 0.02595564, 0.06218609, 0.01457794, 0.00244345]), 'param_optimizer': masked_array(data=['adam', 'adadelta', 'adam', 'adam', 'rmsprop', 'adam',
#                    'adadelta', 'rmsprop', 'rmsprop', 'adadelta'],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.1, 0.3, 0.3, 0.1],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_batch_size': masked_array(data=[40, 30, 10, 20, 40, 10, 10, 50, 20, 50],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 40}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 30}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 20}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 40}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 10}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 20}, {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 50}], 'split0_test_score': array([0.95585001, 0.35049999, 0.95485002, 0.95550001, 0.95854998,
#        0.9544    , 0.48345   , 0.95485002, 0.95020002, 0.21654999]), 'split1_test_score': array([0.96289998, 0.30904999, 0.93945003, 0.94975001, 0.9533    ,
#        0.95109999, 0.49564999, 0.94935   , 0.94980001, 0.19335   ]), 'split2_test_score': array([0.95534998, 0.3495    , 0.95275003, 0.95840001, 0.95635003,
#        0.94405001, 0.43794999, 0.95740002, 0.94515002, 0.1956    ]), 'mean_test_score': array([0.95803332, 0.33634999, 0.94901669, 0.95455001, 0.95606667,
#        0.94985   , 0.47234999, 0.95386668, 0.94838335, 0.20183333]), 'std_test_score': array([0.0034473 , 0.01930833, 0.00681876, 0.00359467, 0.00215264,
#        0.00431682, 0.02482915, 0.00335916, 0.00229214, 0.01044671]), 'rank_test_score': array([ 1,  9,  6,  3,  2,  5,  8,  4,  7, 10])}

# GridSearchCV
