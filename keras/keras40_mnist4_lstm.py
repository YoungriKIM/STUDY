# (N, 28, 28)     CNN
# (N, 764)        Dense
# (N, 764, 1)      LSTM     > input_shape = (28*28, 1) > (28*14, 2) > (28*7, 4) 등이 더 빠를 것이다.
# lstm으로 구성

# 추가로 새 파일 만들어서 boston, diabetes, cancer, iris, wine CNN으로 만들 것