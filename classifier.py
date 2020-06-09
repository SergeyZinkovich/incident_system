import keras
from preprocess_udpipe import done_text

NN_MODEL_FILENAME = 'nn_model/model.h5'


def classify(text):
    model = keras.models.load_model(NN_MODEL_FILENAME)
    y_predicted = model.predict(prepare_text(text))


def prepare_text(text):
    words = done_text(text)
    vec = []
    for w in words:
        if w not in w2v_model.wv:
            continue
        vec.append(w2v_model.wv.get_vector(w))
        if len(vec) > 1000:
            break
    return vec
