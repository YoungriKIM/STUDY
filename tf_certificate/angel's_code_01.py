import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)

    # 모델 - relu 말고, linear사용
    model = Sequential()
    model.add(Dense(128, input_dim =1))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(4))
    model.add(Dense(1, activation='linear')) # 마지막은 linear

    # 컴파일, 훈련
    model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'])
    model.fit(xs, ys, batch_size=1, epochs=200)

    # 평가, 예측 -->model.predict([10.0])을 넣은값 출력
    loss = model.evaluate(xs, ys)
    y_pred = model.predict([10.0])
    print(y_pred) # 11이 나와야 좋은 모델

    # YOUR CODE HERE
    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
