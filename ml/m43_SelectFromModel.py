# feature_importances 의 업그레이드 버전을 써보자!
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html

import numpy as np

from xgboost import XGBClassifier, XGBRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel   # feature_importance와 관련
from sklearn.metrics import r2_score, accuracy_score

# 사전 모델 구성
x, y = load_boston(return_X_y = True) # return_X_y : x,y 바로 분리되어 나옴
# 대신 스플릿은 내가 알아서
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 23)

# 모델을 xgb로 구성
model = XGBRegressor(n_jobs = 8)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('r2 : ', score)
# r2 :  0.9221188601856797


# xgoost 는 feature_importances로 피쳐별 중요도를 확인 할 수 있다.  ------------------------------
thresholds = np.sort(model.feature_importances_)    #오름차순
print(thresholds)   # 13개가 나올 것 이다.
# random_state 값 별로 다르게 나온다.
# [0.00104329 0.00164238 0.01016065 0.01019937 0.01235731 0.01622222
#  0.0170513  0.02586146 0.03516049 0.04727715 0.05181789 0.34372923
#  0.42747724]
print(np.sum(thresholds))   #1.0        # 모두 합쳐서 0 인지 확인해야 한다.

# SelectFromModel을 해보자 --------------------------------------------------------------------
# 중요도 가중치를 기반으로 기능을 선택하기위한 메타 트랜스포머.
# Xgbooster, LGBM, RandomForest등 feature_importances_기능을 쓰는 모델이면 사용 가능
# prefit: True 인 경우 transform직접 호출 
for thresh in thresholds:
    selection = SelectFromModel(model, 
                                threshold = thresh, #Feature 선택에 사용할 임계 값
                                prefit=True         #사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상되는지 여부
                                )

    # x_train을 선택한 Feature로 줄입니다.
    selection_x_train = selection.transform(x_train) # x_train 을 selection 형태로 바꿈
    print(selection_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 8)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(selection_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2:%.2f%%' %(thresh, selection_x_train.shape[1], score*100))

# ==============================
# (404, 13)
# Thresh=0.002, n=13, R2:85.36%
# (404, 12)
# Thresh=0.003, n=12, R2:85.25%
# (404, 11)
# Thresh=0.007, n=11, R2:83.56%
# (404, 10)
# Thresh=0.010, n=10, R2:85.35%     # 피쳐를 3개 줄였지만 값은 0.01밖에 차이가 안 난다.
# (404, 9)
# Thresh=0.014, n=9, R2:83.77%
# (404, 8)
# Thresh=0.021, n=8, R2:84.01%
# (404, 7)
# Thresh=0.024, n=7, R2:82.38%
# (404, 6)
# Thresh=0.034, n=6, R2:84.36%
# (404, 5)
# Thresh=0.036, n=5, R2:79.98%
# (404, 4)
# Thresh=0.040, n=4, R2:78.25%
# (404, 3)
# Thresh=0.051, n=3, R2:71.92%
# (404, 2)
# Thresh=0.194, n=2, R2:66.59%
# (404, 1)
# Thresh=0.564, n=1, R2:43.27%



# m44 기울기, 편향 부분 여기서도 확인
print(model.coef_)
print(model.intercept_)
# 에러 발생
# AttributeError: Coefficients are not defined for Booster type None
# 기울기는 부스터 타입에서는 정의되지 않았다.
# 다만 이름이 다른 같은 기능이 있다. 기울기, 편향이 없는 것이 아니다.