# 오토케라스 깔아서 mnist로 해보기

import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# 원핫인코딩 안 해도 되네..?

# 그래서 원핫인코딩 해부자
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# 그럼 안 돌아가야 하는데? 근데 돌아가네? >> 사이킷런화 해서 여러 모델을 쓸 수 있다. imageclassifier에서는 원핫해도 되고 안 해도 된다

# 인풋 쉐이프가 없네 ?

model = ak.ImageClassifier(
    # overwrite=True,
    max_trials=1    # epoch을 *n번 시도했다.
)

# 컴파일 안 하네?

model.fit(x_train, y_train, epochs=1)

results = model.evaluate(x_test, y_test)

print(results)

# 자동으로 체크 포인트까지 생성이 된다.