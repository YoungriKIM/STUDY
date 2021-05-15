
import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    with open('sarcasm.json') as file:
        data = json.load(file)

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

    for item in data:
        sentences.append(item['headline'])  # 뉴스 기사의 헤드라인
        labels.append(item['is_sarcastic'])  # 뉴스 헤드라인이 Sarcastic하다면 1, 그렇지 않다면 0.

    token = Tokenizer(num_words = vocab_size, oov_token= oov_tok)
    token.fit_on_texts(sentences)
    sentences = token.texts_to_sequences(sentences)
    sentences = pad_sequences(sentences, maxlen = max_length, padding = padding_type, truncating= trunc_type)

    # print(len(sentences)) 26709
    x_train = np.array(sentences[0:training_size])
    x_test = np.array(sentences[training_size:])
    y_train = np.array(labels[0:training_size])
    y_test = np.array(labels[training_size:])

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv1D(64, 5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        # tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.MaxPool1D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    es = EarlyStopping(patience= 8)
    lr = ReduceLROnPlateau(factor = 0.25, patience = 4, verbose = 1)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model.fit(x_train, y_train, epochs = 1000, validation_split = 0.2, callbacks = [es, lr])
    print(model.evaluate(x_test,y_test))

    return model