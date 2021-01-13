import numpy as np
import pandas as pd

# 데이터 불러옴
df = pd.read_csv('../data/csv/ss_data.csv', index_col=0, header=0, encoding='cp949', thousands=',') 
# print(df)

# 데이터 순서 역으로
df2 = df.iloc[::-1].reset_index(drop=True)
# print(df2)  (2400, 14)

print(df2.info())



'''
x = df2.iloc[1740:2400, [0,1,2,4,5,6,7,8,9,10,11,12,13]]
y = df2.iloc[1740:2400, 3]

print(x.shape)      # (660, 13)
print(y.shape)      # (660,)

print(type(x))
print(type(y))

# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.series.Series'>

x = x.to_numpy()
y = y.to_numpy()

print(type(x))      # <class 'numpy.ndarray'>
print(type(y))      # <class 'numpy.ndarray'>



print(x)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=13, activation='relu'))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

loss = model.evaluate(x, y, batch_size=1)
print('loss: ', loss)
'''