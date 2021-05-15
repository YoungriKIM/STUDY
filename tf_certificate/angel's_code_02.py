
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
    # print(y_train.shape, y_test.shape) # (60000,) (10000,)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      train_size=0.8, shuffle=True, random_state=6)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    # 모양은 원래 (60000, 28, 28)이므로 전처리만
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.

    x_train= x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)
    x_val = x_val.reshape(-1, 28, 28)


    # ========= 모델 ==============
    model = Sequential()
    model.add(Conv1D(filters=50, kernel_size=2, padding='same', input_shape = (28, 28)))
    # model.add(Dense(56, input_shape=(28, 28)))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(10, activation='softmax'))

    # ============= 컴파일, 훈련 ==============
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=8, epochs=30)

    # ============= 평가, 예측 ================
    acc = model.evaluate(x_test, y_test, batch_size=8)
    print('acc', acc[1])
    # y_pred = model.predict(x_test)
    # print(y_pred)

    # YOUR CODE HERE
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
