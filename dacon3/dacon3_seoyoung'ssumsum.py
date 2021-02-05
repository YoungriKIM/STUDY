# 서영이가 알려준 거 활용했음 짱짱맨 잘 나온 파일 모아서 최빈값으로 넣는 것

import numpy as np
import pandas as pd
from collections import Counter

# 코랩에서 쓴거 붙여넣음

# from google.colab import drive
# drive.mount('/content/drive')

# 파일 고르는 중

sum1 = pd.read_csv('../data/csv/dacon3/3th_acc1_1.csv')
sum2 = pd.read_csv('../data/csv/dacon3/3th_acc1_2.csv')
sum3 = pd.read_csv('../data/csv/dacon3/3th_acc1_3.csv')

sum1 = sum1.iloc[:,1]
sum2 = sum2.iloc[:,1]
sum3 = sum3.iloc[:,1]

sumsum = pd.concat([sum1, sum2, sum3], axis=1)

print(sumsum.shape[0])

sub = pd.read_csv('../data/csv/dacon3/submission.csv')

for i in range(sumsum.shape[0]) :
    predicts = sumsum.loc[i, : ]
    sub.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]
print(sub.head())
sub = sub[['id', 'digit']]
print(sub.head())
print(sub.shape)
sub.to_csv('../data/csv/dacon3/3th_acc1_sum.csv', index = False)

print('===save complete===')

# =================================================
# 결과 좋은 파일 합 > seoyounglegend.csv > dacon score : 0.9509803922
# 4개로 합친 것 > 0.9509803922 똑같음ㅎ;
# 3th 모델 에큐러시 1 3개 합친 것 > 3th_acc1_sum > dacon score : 0.94