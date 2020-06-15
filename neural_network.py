import numpy as np
import pandas as pd
import keras
import pickle
import matplotlib.pyplot as plt
import configparser
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from preprocess_udpipe import done_text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import plot_model


config = configparser.ConfigParser()
config.read("config.ini")

TRAIN_TEST_SPLIT = int(config["DEFAULT"]["TRAIN_TEST_SPLIT"])
NN_MODEL_FILENAME = config["DEFAULT"]["NN_MODEL_FILENAME"]
X_ARRAY_FILENAME = config["DEFAULT"]["X_ARRAY_FILENAME"]
Y_ARRAY_FILENAME = config["DEFAULT"]["Y_ARRAY_FILENAME"]
THEME_DICT_FILENAME = config["DEFAULT"]["THEME_DICT_FILENAME"]
META_DICT_FILENAME = config["DEFAULT"]["META_DICT_FILENAME"]
TOKENIZER_FILENAME = config["DEFAULT"]["TOKENIZER_FILENAME"]
MODEL_PLOT_FILENAME = config["DEFAULT"]["MODEL_PLOT_FILENAME"]


def get_data(dataset_filename):
    data = pd.read_csv(dataset_filename, encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    return data.sample(frac=1).reset_index(drop=True)


def prepare_data(dataset_filename):
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
    with open(TOKENIZER_FILENAME, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
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


def train(x, y, max_words, theme_dict_len, unic_words_count):
    train_size = int(len(x) * TRAIN_TEST_SPLIT)
    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model = Sequential()
    model.add(keras.layers.Embedding(unic_words_count, max_words))
    model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(theme_dict_len))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file=MODEL_PLOT_FILENAME)

    train_history = model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), class_weight='auto')
    show_history_plot(train_history)
    model.save(NN_MODEL_FILENAME)


def test(x, y):
    train_size = int(len(x) * TRAIN_TEST_SPLIT)
    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    model = keras.models.load_model(NN_MODEL_FILENAME)

    y_predicted = model.predict(x_test)
    correct_count = 0
    confidences = []
    theme_result_dict = {}
    predictions_count = 0
    for i in range(len(y_predicted)):
        nlargest_id = y_predicted[i].argsort()[-3:]
        max_predicted_index = nlargest_id[2]
        max_real_index = np.argmax(y_test[i])

        if max_real_index not in theme_result_dict:
            theme_result_dict[max_real_index] = [0, 0, 0]

        confidence = abs(y_predicted[i][nlargest_id[2]] - y_predicted[i][nlargest_id[1]])

        if max_predicted_index == max_real_index:
            confidences.append([confidence, True])
            correct_count += 1
            predictions_count += 1
            theme_result_dict[max_real_index][0] += 1
        else:
            confidences.append([confidence, False])
            predictions_count += 1
            theme_result_dict[max_real_index][1] += 1

    for i in y_train:
        theme_result_dict[np.argmax(i)][2] += 1
    print('Accuracy for themes (theme id/train samples count/accuracy):')
    for i in theme_result_dict.keys():
        print(i, ':', theme_result_dict[i][2],
              theme_result_dict[i][0] / (theme_result_dict[i][0] + theme_result_dict[i][1]))

    confidences.sort(key=lambda a: a[0])
    print('Confidences (confidence/result)')
    for i in confidences:
        print(i)

    accuracy = correct_count / predictions_count
    print('Manual test accuracy check: ', accuracy)

    score = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('NN test accuracy: ', score[1])

    model = keras.models.load_model(NN_MODEL_FILENAME)
    score = model.evaluate(x, y, batch_size=128, verbose=1)
    print('NN total accuracy: ', score[1])


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
