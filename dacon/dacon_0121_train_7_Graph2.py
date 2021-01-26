# 예측값으로 그래프 그리기

# from google.colab import drive
# drive.mount('/content/drive')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pandas import DataFrame

pred = pd.read_csv('/content/drive/MyDrive/colab_data/dacon1/dongdong/result.csv')
pred2 = pd.read_csv('/content/drive/MyDrive/colab_data/dacon1/sub_0126_2.csv')

ranges = 1008    #7일치로 보겠음
hours = range(ranges)
pred = pred[ranges+ranges:ranges+ranges+ranges]

q_01 = pred['q_0.1'].values
q_02 = pred['q_0.2'].values
q_03 = pred['q_0.3'].values
q_04 = pred['q_0.4'].values
q_05 = pred['q_0.5'].values
q_06 = pred['q_0.6'].values
q_07 = pred['q_0.7'].values
q_08 = pred['q_0.8'].values
q_09 = pred['q_0.9'].values

hours = range(ranges)
pred2 = pred2[ranges+ranges:ranges+ranges+ranges]

q_01_2 = pred2['q_0.1'].values
q_02_2 = pred2['q_0.2'].values
q_03_2 = pred2['q_0.3'].values
q_04_2 = pred2['q_0.4'].values
q_05_2 = pred2['q_0.5'].values
q_06_2 = pred2['q_0.6'].values
q_07_2 = pred2['q_0.7'].values
q_08_2 = pred2['q_0.8'].values
q_09_2 = pred2['q_0.9'].values

import matplotlib.pyplot as plt

plt.figure(figsize=(36,10))
plt.subplot(2,1,1)
plt.plot(hours, q_01, color='red')
plt.plot(hours, q_02, color='#aa00cc')
plt.plot(hours, q_03, color='#00ccaa')
plt.plot(hours, q_04, color='#ccaa00')
plt.plot(hours, q_05, color='#00aacc')
plt.plot(hours, q_06, color='#aacc00')
plt.plot(hours, q_07, color='#cc00aa')
plt.plot(hours, q_08, color='#000000')
plt.plot(hours, q_09, color='blue')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hours, q_01_2, color='red')
plt.plot(hours, q_02_2, color='#aa00cc')
plt.plot(hours, q_03_2, color='#00ccaa')
plt.plot(hours, q_04_2, color='#ccaa00')
plt.plot(hours, q_05_2, color='#00aacc')
plt.plot(hours, q_06_2, color='#aacc00')
plt.plot(hours, q_07_2, color='#cc00aa')
plt.plot(hours, q_08_2, color='#000000')
plt.plot(hours, q_09_2, color='blue')
plt.legend()

plt.show()
