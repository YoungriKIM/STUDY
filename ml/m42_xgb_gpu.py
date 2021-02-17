# xgb를 cpu가 아닌 gpu로 써보자 ! 대신 cuda가 10.1 이상이어야 한다.

import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=311)

model = XGBRegressor(n_estimators=100000, leaning_rate=0.01\
                     , tree_method='gpu_hist'       # 이 파라미터를 넣으면 gpu로 돌리기 가능!
                    #  , predictor='gpu_predictor'    # 프레딕트할 때(score부분) gpu로 하겠다. 문서상으로는 그런데, 사실상 다 쓰인다.
                     , predictor='cpu_predictor'
                     , gpu_id=0         # gpu가 여러개이면 이렇게 선택해서 쓸 수 있다.
)

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse']\
          , eval_set = [(x_train, y_train), (x_test, y_test)]
          , early_stopping_rounds=10000
)

aaa = model.score(x_test, y_test)
print('model.score: ', aaa)
