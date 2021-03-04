# IMDB movie review sentiment classification dataset : 2진 분류인 감정분류
# 실습해라~
# 임베딩 안 쓴것 까지 해서 비교해라

from tensorflow.keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train.shape, x_test.shape)      # (25000,) (25000,)
print(y_train.shape, y_test.shape)      # (25000,) (25000,)

# 가장 긴 길이를 찾아보자
print('뉴스기사 최대 길이: ', max(len(l) for l in x_train))
print('뉴스기사 평균 길이: ', sum(map(len, x_train))/ len(x_train))

# 그래프로 한 번 그려보자
# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()
# 패딩 길이 600으로 하자
print('==============================')

# y 분포를 확인해보자
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print('y 분포: ', dict(zip(unique_elements, counts_elements)))
# y 분포:  {0: 12500, 1: 12500}
print('==============================')

# 전처리
# 패딩(길이는 아까 그래프보고 확인한 500으로 하자)
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 600
x_train = pad_sequences(x_train, padding='pre', maxlen=max_len)
x_test = pad_sequences(x_test, padding='pre', maxlen=max_len)

# y 벡터화
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 이쯤에서 쉐잎 한 번 확인
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (25000, 600)
# (25000, 600)
# (25000, 2)
# (25000, 2)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=600))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(2,activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
stop = EarlyStopping(monitor='loss', patience=8, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.5)

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1, callbacks=[stop, reduce_lr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss: ', loss)

# ==========================================
# 85_imdb
# loss:  [1.3221393823623657, 0.8501999974250793]

print('done')