import keras
import numpy as np
import pickle
from keras.preprocessing import sequence
from preprocess_udpipe import done_text

NN_MODEL_FILENAME = 'nn_model/model.h5'
TRAINED_MODEL_FILENAME = 'models/trained_model.model'
META_DICT_FILENAME = 'nn_model/meta_dict.pkl'
THEME_DICT_FILENAME = 'nn_model/theme_dict.pkl'
TOKENIZER_FILENAME = 'nn_model/tokenizer.pkl'


def load_tokenizer():
    with open(TOKENIZER_FILENAME, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def load_metadata():
    with open(META_DICT_FILENAME, 'rb') as f:
        meta_dict = pickle.load(f)
    return meta_dict['max_words'], meta_dict['unic_words_count']


def load_theme_dict():
    with open(THEME_DICT_FILENAME, 'rb') as f:
        theme_dict = pickle.load(f)
    return dict((v, k) for k, v in theme_dict.items())


def classify(text):
    model = keras.models.load_model(NN_MODEL_FILENAME)
    theme_dict = load_theme_dict()
    y_predicted = model.predict(prepare_text(text))[0]
    return theme_dict[np.argmax(y_predicted)]


def prepare_text(text):
    tokenizer = load_tokenizer()
    max_words, unic_words_count = load_metadata()
    x = tokenizer.texts_to_sequences([done_text(text)])

    if len(x[0]) > 1000:
        x[0] = x[0][:1000]

    x = sequence.pad_sequences(x, maxlen=max_words)[0]

    x = np.array([x])
    return x


print(classify('"Ошибка проверки возможности регистрации комплекта" Добрый день. 1262571 Сим карта Мегафон с ТП "Включайся! Общайся", (Красноярский край) промо 897010230375489051 +7-923-344-17-25 При попытке зарегистрировать выдает ошибку "Ошибка проверки возможности регистрации комплекта"'))
