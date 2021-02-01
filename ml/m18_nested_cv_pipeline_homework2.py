# wine을 데이터로 randomforest를 써서 
# 파이프라인 엮어서 25번 돌리자! 

import numpy as np
from sklearn.datasets import load_wine
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

dataset = load_wine()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits = 5, shuffle=True)

parameters = [
    {'randomforestclassifier__n_estimators' : [50,100,200]},
    {'randomforestclassifier__max_depth' : [6,8,10,12]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__n_jobs' : [-1,2,4,8]}
]


# 5*5=25 되도록 모델 구성 kfold 먼저 하고 거기서 또 split 되도록
pipemodel = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = RandomizedSearchCV(pipemodel, parameters, cv = kfold)
score = cross_val_score(model, x, y, cv = kfold)

print('nested score: ', score)

#==============================
# m18 nested+pipeline
# nested score:  [0.94444444 1.         1.         0.97142857 1.        ]