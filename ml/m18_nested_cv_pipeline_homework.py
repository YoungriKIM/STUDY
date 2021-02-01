# diabetes을 데이터로 randomforest를 써서 
# 파이프라인 엮어서 25번 돌리자! 
# m17 그대로 하지 말고..여러가지를 시도해야 한다.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline

#classfier = 분류모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits = 5, shuffle=True)

parameters = [
    {'randomforestregressor__n_estimators' : [100,200],
    'randomforestregressor__max_depth' : [6,8,10,12],
    'randomforestregressor__min_samples_leaf' : [3,5,7,10],
    'randomforestregressor__min_samples_split' : [2,3,5,10],
    'randomforestregressor__n_jobs' : [-1]} 
]

# 5*5=25 되도록 모델 구성 kfold 먼저 하고 거기서 또 split 되도록
pipemodel = make_pipeline(MinMaxScaler(), RandomForestRegressor())
model = RandomizedSearchCV(pipemodel, parameters, cv = kfold)
score = cross_val_score(model, x, y, cv = kfold)

print('nested score: ', score)

#==============================
# nested score:  [0.41288673 0.41056633 0.57151867 0.4119841  0.2535303 ]