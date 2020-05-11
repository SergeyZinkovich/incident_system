import numpy as np
import pandas as pd
import keras
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from preprocess_udpipe import done_text


TRAIN_TEST_SPLIT = 0.8
NN_MODEL_FILENAME = 'nn_model/model.h5'
X_ARRAY_FILENAME = 'nn_model/x_array.npy'
Y_ARRAY_FILENAME = 'nn_model/y_array.npy'
THEME_DICT_FILENAME = 'nn_model/theme_dict.pkl'


def get_data(dataset_filename):
    data = pd.read_csv(dataset_filename, encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    return data.sample(frac=1).reset_index(drop=True)


def prepare_data(dataset_filename, w2v_model, dct, tfidf):
    data = get_data(dataset_filename)
    theme_dict = {}
    x = []
    y = []
    j = -1

    for i in data.values:
        if not i[0] in theme_dict:
            j += 1
            theme_dict[i[0]] = j
        words = done_text(str(i[1]))
        vec = 0
        for w in words:
            if w not in w2v_model.wv:
                continue
            v = w2v_model.wv.get_vector(w)
            if vec == 0:
                vec = [v * tfidf.idfs[dct.token2id[w]], 1]
            else:
                vec = [vec[0] + v * tfidf.idfs[dct.token2id[w]], vec[1] + 1]
        vec = vec[0] / vec[1]
        x.append(vec)
        y.append(theme_dict[i[0]])

    x = np.array(x)
    y = np.array(y)
    np.save(X_ARRAY_FILENAME, x)
    np.save(Y_ARRAY_FILENAME, y)
    with open(THEME_DICT_FILENAME, 'wb') as f:
        pickle.dump(theme_dict, f, pickle.HIGHEST_PROTOCOL)
    return x, y, theme_dict


def load_prepared_data():
    x = np.load(X_ARRAY_FILENAME)
    y = np.load(Y_ARRAY_FILENAME)
    with open(THEME_DICT_FILENAME, 'rb') as f:
        theme_dict = pickle.load(f)
    return x, y, theme_dict


def train(w2v_model_vector_size, x, y, theme_dict_len):
    train_size = int(len(x) * TRAIN_TEST_SPLIT)
    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model = Sequential()
    model.add(Dense(256, input_shape=(w2v_model_vector_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(theme_dict_len))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    train_history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test), class_weight='auto')
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
