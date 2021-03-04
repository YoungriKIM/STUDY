# 샘이 같이 해준거

from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# !! 로이터 데이터셋이란? : This is a dataset of 11,228 newswires from Reuters, labeled over 46 topics.
# 46개의 토픽의 라벨을 갖고 있는 11.284개의 뉴스 속 문장 데이터이다.

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)
# reuters > 영국의 로이터 통신사
# num_word = 처음부터 불러올 단어의 수를 지정할 수 있다!

print(x_train[0], type(x_train[0]))         # [1, 2, 2 ··· , 15, 17, 12] ,  <class 'list'>    # 정수 인코딩까지 리스트로 들어있다.
print(len(x_train[0]), len(x_train[11]))    # 87 59 > 문장의 길이가 모두 다르다
print(y_train[0])                           # 3
print('==============================')
print(x_train.shape, x_test.shape)      # (8982,) (2246,)
print(y_train.shape, y_test.shape)      # (8982,) (2246,)

# 가장 긴 길이를 찾아보자
print('뉴스기사 최대 길이: ', max(len(l) for l in x_train))
print('뉴스기사 평균 길이: ', sum(map(len, x_train))/ len(x_train))
# 뉴스기사 최대 길이:  2376
# 뉴스기사 평균 길이:  145.5398574927633

# 그래프로 한 번 그려보자
# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()
# 패딩 길이는 600으로 하면 될 것 같다.
print('==============================')


# y 분포를 확인해보자
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print('y 분포: ', dict(zip(unique_elements, counts_elements)))
# y 분포:  {0: 55, 1: 432, 2: 74, ··· , 43: 21, 44: 12, 45: 18}
# 0이 55번, 1이 432번 있다는 의미

# 그래프로 한 번 그려보자
# plt.hist(y_train, bins=46)
# plt.show()
print('==============================')


# x 단어 분포를 확인해보자
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
# {'mdbl': 10996, 'fawc': 16260, ··· , 'oversight': 13843, "paradyne's": 20814}
print('==============================')


# x에 부여된 정수와 단어의 위치를 바꿔서 원래 어떤 문장인지 보자!
# 키와 밸류를 바꾸면 되겠지?
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key
# 키 밸류 교환 후
print(index_to_word)
print(index_to_word[1])
print(len(index_to_word))
print(index_to_word[30979])
# {10996: 'mdbl', 16260: 'fawc', ··· , 13843': 'oversight', 20814: "paradyne's"}
# the               # 정수를 1로 부여받은 가장 많이 나온 단어이다.
# 30979
# northerly         # 정수를 크게 받은 가장 적게 나온 단어이다.
print('==============================')

# 문장으로 복원하자(num_word=10000 일 때)
print(x_train[0])
# [1, 2, 2 ··· , 15, 17, 12]
print(' '.join([index_to_word[index] for index in x_train[0]]))
# the of of mln loss for plc said at only ended said of could 1 traders now april 0 a after said from 1985 and from foreign 000 april\
# 0 prices its account year a but in this mln home an states earlier and rise and revs vs 000 its 16 vs 000 a but 3 of of several\
# and shareholders and dividend vs 000 its all 4 vs 000 1 mln agreed of april 0 are 2 states will billion total and against 000 pct dlrs
print('==============================')

# y 카테고리 갯수 출력
category = np.max(y_train) + 1
print('y 카테고리의 개수: ', category)
# y 카테고리의 개수:  46

# y 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
<<<<<<< HEAD
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
=======
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
print('==============================')

# 전처리 해주자
# 패딩(길이는 아까 그래프보고 확인한 500으로 하자)
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')
print(x_train.shape, x_test.shape)
# (8982, 100) (2246, 100)

# y 벡터화
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)
# (8982, 46) (2246, 46)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))
model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# y 카테고리컬을 끄고! sparse_categorical_crossentropy 을 써보자
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 이제 다중 분류할 때 쓸 수 있는 로스가 하나 더 생겼다!

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

result = model.evaluate(x_test, y_test)
print(result)

# ===================================
# categorical_crossentropy
# [1.493270993232727, 0.6682991981506348]
# sparse_categorical_crossentropy
# [1.57481050491333, 0.6513802409172058]

>>>>>>> 080954d50305513d288041d53afeec10d7d1e0fc
