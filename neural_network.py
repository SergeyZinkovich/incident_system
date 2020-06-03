import numpy as np
import pandas as pd
import keras
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from preprocess_udpipe import done_text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import plot_model


TRAIN_TEST_SPLIT = 0.8
NN_MODEL_FILENAME = 'nn_model/model.h5'
X_ARRAY_FILENAME = 'nn_model/x_array.npy'
Y_ARRAY_FILENAME = 'nn_model/y_array.npy'
THEME_DICT_FILENAME = 'nn_model/theme_dict.pkl'
META_DICT_FILENAME = 'nn_model/meta_dict.pkl'


def get_data(dataset_filename):
    data = pd.read_csv(dataset_filename, encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    return data.sample(frac=1).reset_index(drop=True)


def prepare_data(dataset_filename, w2v_model, dct, tfidf):
    data = get_data(dataset_filename)
    theme_dict = {}
    y = []
    j = -1
    max_words = 0

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(map(done_text, data.text.tolist()))

    x = tokenizer.texts_to_sequences(map(done_text, data.text.tolist()))

    for i in range(len(x)):
        if len(x[i]) > 1000:
            x[i] = x[i][:1000]
        if len(x[i]) > max_words:
            max_words = len(x[i])

    x = sequence.pad_sequences(x, maxlen=max_words)

    for id in data.id:
        if id not in theme_dict:
            j += 1
            theme_dict[id] = j
        y.append(theme_dict[id])
    y = keras.utils.to_categorical(y, len(theme_dict))

    x = np.array(x)
    y = np.array(y)

    meta_dict = {
        'max_words': max_words,
        'unic_words_count': len(tokenizer.word_counts)
    }

    np.save(X_ARRAY_FILENAME, x)
    np.save(Y_ARRAY_FILENAME, y)
    with open(THEME_DICT_FILENAME, 'wb') as f:
        pickle.dump(theme_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(META_DICT_FILENAME, 'wb') as f:
        pickle.dump(meta_dict, f, pickle.HIGHEST_PROTOCOL)

    return x, y, theme_dict, max_words, len(tokenizer.word_counts)


def load_prepared_data():
    x = np.load(X_ARRAY_FILENAME)
    y = np.load(Y_ARRAY_FILENAME)
    with open(THEME_DICT_FILENAME, 'rb') as f:
        theme_dict = pickle.load(f)
    with open(META_DICT_FILENAME, 'rb') as f:
        meta_dict = pickle.load(f)
    return x, y, theme_dict, meta_dict['max_words'], meta_dict['unic_words_count']


def train(w2v_model_vector_size, x, y, max_words, theme_dict_len, unic_words_count):
    train_size = int(len(x) * TRAIN_TEST_SPLIT)
    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model = Sequential()
    model.add(keras.layers.Embedding(unic_words_count, max_words))
    model.add(keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(theme_dict_len))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model.png')

    train_history = model.fit(x_train, y_train, batch_size=128, epochs=200, validation_data=(x_test, y_test), class_weight='auto')
    show_history_plot(train_history)
    model.save(NN_MODEL_FILENAME)


def test(x, y):
    train_size = int(len(x) * TRAIN_TEST_SPLIT)
    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model = keras.models.load_model(NN_MODEL_FILENAME)
    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('test accuracy: ', score[1])

    model = keras.models.load_model(NN_MODEL_FILENAME)
    score = model.evaluate(x, y, batch_size=128, verbose=1)
    print('total accuracy: ', score[1])


def show_history_plot(train_history):
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
