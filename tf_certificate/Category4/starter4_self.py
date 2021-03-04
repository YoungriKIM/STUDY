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
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    # url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    # urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE

    # load_json
    import pandas as pd
    df = pd.read_json ('../Study/tf_certificate/Category4/sarcasm.json', orient ='index ')

    print(df.head())

    # x, y 지정
    pre_x = df['headline']
    y = df['is_sarcastic']

    print(pre_x.shape)
    print(y.shape)
    # (26709,)
    # (26709,)
    # 26709개의 문장이 있다.

    # 미리 주어준 변수
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    # 전처리
    # 정수 인코딩
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(vocab_size, oov_tok)
    tokenizer.fit_on_texts(pre_x)
    x = tokenizer.texts_to_sequences(pre_x)

    # 패딩
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    x = pad_sequences(x, maxlen = max_length, padding=padding_type, truncating=trunc_type)

    # 스플릿
    x_train = x[0:training_size]
    x_test = x[training_size:]
    y_train = y[0:training_size]
    y_test = y[training_size:]

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # (20000, 120)
    # (6709, 120)
    # (20000,)
    # (6709,)

    from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(128)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 컴파일, 훈련
    stop = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=8)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[stop])

    # 평가
    loss = model.evaluate(x_test, y_test)
    print(loss)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")


# [0.3825017511844635, 0.827843189239502]
