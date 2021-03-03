from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer() # 토큰처럼 자를거야!
token.fit_on_texts([text])

print(token.word_index)
# {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}
# 워드 인덱스의 값은 1부터 시작한다.

x = token.texts_to_sequences([text])
print(x)
# [[3, 1, 1, 4, 5, 1, 2, 2, 6]]

# 벡터화를 해야겠지~
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print('단어의 개수: ', word_size)    # 단어의 개수:  6

x = to_categorical(x)
print(x)
# 워드 인덱스의 값은 1부터 시작한다.
# 그래서 0번으로 지정된 것이 없다. 알아서 지정하던지!
# [[[0. 0. 0. 1. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1.]]]
print(x.shape) # (1, 9, 7)
