# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # (60000, 28, 28)
    # (10000, 28, 28)
    # (60000,)
    # (10000,)

    # 전처리
    # val 나누기
    # /255.
    # y 벡터화

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)

    x_train = x_train.astype('float32')/255.
    x_val = x_val.astype('float32')/255.
    x_test = x_test.astype('float32')/255.

    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)


    # 모델 구성
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, strides=1, padding='same', input_shape=(28,28)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32))
    model.add(Dense(32))
    model.add(Dense(10, activation='softmax'))

    # 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=16, mode='auto')

    model.fit(x_train, y_train, epochs=30, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])

    # 평가
    loss = model.evaluate(x_test, y_test, batch_size=32)
    print('loss, acc: ', loss)
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")


# =====================================================
# loss, acc:  [0.43111270666122437, 0.8730999827384949]