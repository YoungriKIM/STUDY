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

search = GridSearchCV(model2, hyperparameters, cv = 3)
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
# cv_result:  {'mean_fit_time': array([9.21584336, 7.88297272, 7.88993947, 9.12637353, 7.92320466,
#        7.8627069 , 9.08140016, 8.03970257, 7.82777731, 5.09049749,
#        4.49639066, 4.68882465, 5.12000299, 4.55305235, 4.67442656,
#        5.15281781, 4.48449747, 4.38681149, 4.43331885, 3.94329325,
#        3.79673696, 4.62987137, 3.73277839, 3.61041069, 4.11487921,
#        3.87497854, 3.73920616, 3.14323632, 2.60366694, 2.60750024,
#        2.98677437, 2.60155177, 2.67523805, 2.95952241, 2.44936172,
#        2.71161   , 2.67158   , 2.21361009, 2.16616074, 2.56646029,
#        2.16181819, 2.28798421, 2.64684439, 2.06103253, 2.39045453]), 'std_fit_time': array([0.34346475, 0.07258992, 0.11498214, 0.19401724, 0.18048424,
#        0.21137031, 0.1846354 , 0.25808908, 0.09672623, 0.1841203 ,
#        0.09830143, 0.23064692, 0.05223591, 0.05840849, 0.27205411,
#        0.17316287, 0.0932945 , 0.03283759, 0.14171059, 0.13517413,
#        0.15578525, 0.36898237, 0.01921425, 0.02134677, 0.01263986,
#        0.17708471, 0.13502561, 0.14030618, 0.17159878, 0.15050297,
#        0.16708842, 0.19215086, 0.13491357, 0.1546058 , 0.03842473,
#        0.25944614, 0.20343824, 0.19233815, 0.10819171, 0.08267259,
#        0.08388701, 0.16535838, 0.2097761 , 0.02919997, 0.10412114]), 'mean_score_time': array([2.82353417, 2.80810889, 2.79019006, 2.78493015, 2.94889363,
#        2.77983665, 2.77949158, 2.81514907, 2.84271828, 1.47949894,
#        1.56211456, 1.57154322, 1.47302739, 1.54960632, 1.62140203,
#        1.54806844, 1.56600571, 1.5636123 , 1.35176436, 1.30697179,
#        1.31191254, 1.31084911, 1.41750924, 1.31881356, 1.38550536,
#        1.29584837, 1.3057518 , 0.91348513, 0.88816468, 0.89169224,
#        0.88419596, 0.85364946, 0.83211581, 0.84806013, 1.05107061,
#        0.89771708, 0.75437848, 0.74081953, 0.74163302, 0.90663711,
#        0.7710642 , 0.73221318, 0.72015508, 0.75619992, 0.75771928]), 'std_score_time': array([0.11233609, 0.06855405, 0.10442722, 0.07205977, 0.27144798,
#        0.0437855 , 0.06757103, 0.08744842, 0.07968248, 0.04827849,
#        0.06284359, 0.11146894, 0.04612855, 0.0988889 , 0.16561237,
#        0.0790636 , 0.08073291, 0.08728545, 0.06906368, 0.00735894,
#        0.00816282, 0.0251685 , 0.18318689, 0.01944626, 0.14103416,
#        0.01974851, 0.01526542, 0.04784364, 0.04520903, 0.02814921,
#        0.04173029, 0.01382501, 0.0015368 , 0.04042893, 0.18357225,
#        0.04586824, 0.03680762, 0.04281567, 0.03290214, 0.15710614,
#        0.02093126, 0.02227617, 0.00956446, 0.03609093, 0.05328884]), 'param_batch_size': masked_array(data=[10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20,
#                    20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40,
#                    40, 40, 40, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50,
#                    50, 50, 50],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1,
#                    0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2,
#                    0.2, 0.2, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2,
#                    0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3,
#                    0.3],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_optimizer': masked_array(data=['rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
#                    'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
#                    'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta',
#                    'rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
#                    'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
#                    'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta',
#                    'rmsprop', 'adam', 'adadelta', 'rmsprop', 'adam',
#                    'adadelta', 'rmsprop', 'adam', 'adadelta', 'rmsprop',
#                    'adam', 'adadelta', 'rmsprop', 'adam', 'adadelta'],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'batch_size': 10, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 10, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 20, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 30, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 40, 'drop': 0.3, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 
# 0.1, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.1, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.2, 'optimizer': 'adam'}, {'batch_size': 50, 
# 'drop': 0.2, 'optimizer': 'adadelta'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'rmsprop'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'adam'}, {'batch_size': 50, 'drop': 0.3, 'optimizer': 'adadelta'}], 'split0_test_score': array([0.95534998, 0.94954997, 0.46134999, 0.94325   , 0.95625001,
#        0.315     , 0.94279999, 0.95420003, 0.23355   , 0.95674998,
#        0.95789999, 0.35505   , 0.95674998, 0.95420003, 0.28049999,
#        0.95725   , 0.95475   , 0.21475001, 0.94494998, 0.95660001,
#        0.28670001, 0.95394999, 0.95069999, 0.30145001, 0.95415002,
#        0.95899999, 0.26225001, 0.95850003, 0.96065003, 0.23185   ,
#        0.95130002, 0.9558    , 0.19645   , 0.95975   , 0.95885003,
#        0.24529999, 0.95660001, 0.95779997, 0.22724999, 0.95835   ,
#        0.95959997, 0.21905001, 0.95700002, 0.95749998, 0.15790001]), 'split1_test_score': array([0.9418    , 0.95560002, 0.49555001, 0.95405   , 0.95585001,
#        0.39500001, 0.94220001, 0.95284998, 0.3125    , 0.95625001,
#        0.95410001, 0.33485001, 0.95310003, 0.95120001, 0.30019999,
#        0.94999999, 0.95139998, 0.16509999, 0.9569    , 0.95370001,
#        0.23005   , 0.94575   , 0.95475   , 0.26615   , 0.94344997,
#        0.95279998, 0.27135   , 0.95964998, 0.96015   , 0.28995001,
#        0.95455003, 0.96030003, 0.14094999, 0.95415002, 0.95569998,
#        0.1191    , 0.94564998, 0.95740002, 0.13824999, 0.95639998,
#        0.95539999, 0.20995   , 0.95204997, 0.94975001, 0.1067    ]), 'split2_test_score': array([0.95295   , 0.958     , 0.47170001, 0.94685   , 0.95475   ,
#        0.38644999, 0.95469999, 0.95359999, 0.31185001, 0.95880002,
#        0.95635003, 0.28244999, 0.9544    , 0.95660001, 0.16814999,
#        0.95359999, 0.95684999, 0.27065   , 0.95429999, 0.95039999,
#        0.31169999, 0.95875001, 0.95639998, 0.1666    , 0.95504999,
#        0.95060003, 0.21175   , 0.95924997, 0.95130002, 0.20835   ,
#        0.95700002, 0.95384997, 0.20385   , 0.95719999, 0.95525002,
#        0.2348    , 0.95674998, 0.95749998, 0.29225001, 0.96004999,
#        0.95835   , 0.20215   , 0.9508    , 0.95434999, 0.21465001]), 'mean_test_score': array([0.95003333, 0.95438333, 0.4762    , 0.94805   , 0.95561667,
#        0.36548333, 0.94656666, 0.95355   , 0.28596667, 0.95726667,
#        0.95611668, 0.32411667, 0.95475   , 0.95400002, 0.24961666,
#        0.95361666, 0.95433333, 0.21683333, 0.95204999, 0.95356667,
#        0.27615   , 0.95281667, 0.95394999, 0.24473334, 0.95088333,
#        0.95413333, 0.24845   , 0.95913333, 0.95736669, 0.24338334,
#        0.95428336, 0.95665   , 0.18041666, 0.95703334, 0.95660001,
#        0.19973333, 0.95299999, 0.95756666, 0.21925   , 0.95826666,
#        0.95778332, 0.21038334, 0.95328333, 0.95386666, 0.15975   ]), 'std_test_score': array([0.00590371, 0.00355537, 0.01432009, 0.00448999, 0.00063421,
#        0.03586736, 0.00575635, 0.00055229, 0.03706513, 0.00110329,
#        0.00156008, 0.03059514, 0.0015105 , 0.00220907, 0.05816434,
#        0.00295983, 0.00224438, 0.04311578, 0.00513144, 0.0025329 ,
#        0.03415804, 0.00536739, 0.00239479, 0.05709721, 0.005269  ,
#        0.00355652, 0.02621539, 0.00047666, 0.00429463, 0.03429678,
#        0.00233464, 0.00270095, 0.02807019, 0.00228922, 0.00160157,
#        0.05717729, 0.0051976 , 0.00016995, 0.06312422, 0.00149128,
#        0.00176083, 0.0069062 , 0.00267718, 0.00318232, 0.04408981]), 'rank_test_score': array([28, 13, 31, 29, 11, 32, 30, 22, 34,  6, 10, 33, 12, 17, 36, 20, 14,
#        41, 26, 21, 35, 25, 18, 38, 27, 16, 37,  1,  5, 39, 15,  8, 44,  7,
#         9, 43, 24,  4, 40,  2,  3, 42, 23, 19, 45])}