from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재미있어요', '참 최고에요', '갸꿀이에요', '참 너무 잘 만든 영화에요.','추천하고 싶은 영화입니다.', \
        '한 번 더 보고싶네요', '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요',\
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘생기긴 했어요']

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있어요': 3, '최고에요': 4, '갸꿀이에요': 5, '잘': 6, '만든': 7, '영화에요': 8,\
#  '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고싶네요': 15, '글쎄요': 16,\
#  '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, \
# '재미없다': 23, '재밌네요': 24, '규현이가': 25, '잘생기긴': 26, '했어요': 27}
# 정수가 부여되었음을 확인

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [5], [1, 6, 7, 8], [9, 10, 11], [12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27]]

# 이제 패딩으로 길이를 맞춰줄 건데
# lstm의 경우 뒷부분에 있는 정보가 중요해지기 때문에 0을 앞으로 채워주는 것이 좋다.
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre')
# post 는 뒤로 채워진다.
# 길이를 지정하지 않으면 알아서 가장 큰 길이로 하는데, 내가 정하고 싶다면? maxlen= 에 지정하면 되겠지.
# 길 경우 어느 부분이 잘릴까? 기본값으로 앞이 잘린다. truncating='pre'
print(pad_x)
#  [11 12 13 14 15] > maxlen 4로 해서 >  [12 13 14 15]
print(pad_x.shape)
# (14, 5)

# 이 상태에서 모델에 널을 수 있을까?
# lstm 에 (14,5,1) / dense에 input_shape=(5,) 로 가능할 거야

# 유니크한 값을 보자
print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27] / 27+1(0번째 패딩) = 28개 인데 maxlen 4하면서 11은 없을 것이다.

# 모델에 들어가기 전에 벡터화를 해줘야 하잖아
# 근데 원핫벡터는 단어의 종류가 많아지면 길이가 너무 길어지니까 2차원에 뿌려주고 싶어 그걸 'Embedding' 레이어에서 해결할 수 있다.

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
# model.add(Embedding(28, 11))
# > model.add(Embedding(28, 11)) 이것도 된다. input_length 명시 안하면 (None, None, 11) 으로 들어가면서 자동으로 명시된다.
# 이 둘의 파라미터는 308로 같다. 이는 총 단어의 수 * 내가 지정한 아웃풋 딤 = 28 * 11 = 308이다.
# 즉 input_length는 안 지정해줘도 된다~!
# -------------------------------------------------
# input_dim=28 > 총 단어의 개수     > 총 단어의 수보다 작아지면 안 돌아가고, 크면 연산수만 늘어나고 돌아간다.
# output_dim=11 > 내가 원하는 아웃풋 dim (벡터화의 크기)
# input_length=5 > 내가 넣을 문장들의 최대 길이
# -------------------------------------------------
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print('acc: ', acc)

# ===================
# 83_1
# acc:  1.0