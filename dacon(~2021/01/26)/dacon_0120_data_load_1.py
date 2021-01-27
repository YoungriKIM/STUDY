# 81개의 test의 데이터 불러와서 합치기

import numpy as np
import pandas as pd

def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[:,[1,3,4,5,6,7,8]]

df_test = []

for i in range(81):
    file_path = '../data/csv/dacon1/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)

all_test = pd.concat(df_test)
#Attach padding dummy time series
# X_test = X_test.append(X_test[-96:])
print(all_test.shape)     #(27216, 7) > 81,336,7




#--------------------------------------
# 예측값을 제출 형식에 넣기 (예측한 값 10컬럼에 다 복붙함)
sub = pd.read_csv('../data/DACON_0126/sample_submission.csv')

for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day7"), column_name] = y_predict[:,0]
for i in range(1,10):
    column_name = 'q_0.' + str(i)
    sub.loc[sub.id.str.contains("Day8"), column_name] = y_predict[:,1]

# sub.to_csv('../data/DACON_0126/submission/submission_0119_1.csv', index=False)    
# sub.to_csv('../data/DACON_0126/submission/submission_0119_2.csv', index=False)    # score : 2.8878722584
sub.to_csv('../data/DACON_0126/submission/submission_0119_3.csv', index=False)      # score : 3.2707340024