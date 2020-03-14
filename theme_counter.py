import pandas as pd
import gensim
import gensim.models
from gensim.utils import simple_preprocess
from preprocess_udpipe import done_text
import numpy as np
import pickle

def get_data():
    data = pd.read_csv('file.csv', encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    #data = [str(d) for d in data.text]
    return data #map(simple_preprocess, data)

def get_model():
    return gensim.models.Word2Vec.load('models/my_model.model')

def save_obj(obj):
    with open('obj/'+ "theme_dict" + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj():
    with open('obj/' + "theme_dict" + '.pkl', 'rb') as f:
        return pickle.load(f)

def count_theme_dict():
    model = get_model()
    theme_dict = {}
    for i in get_data().values:
        words = done_text(str(i[1]))
        for w in words:
            vec = model.wv.get_vector(w)
            if not i[0] in theme_dict:
                theme_dict[i[0]] = [vec, 1]
            else:
                theme_dict[i[0]] = [theme_dict[i[0]][0] + vec, theme_dict[i[0]][1] + 1]

    for key, val in theme_dict.items():
        theme_dict[key] = val[0] / val[1]
        a=1
    save_obj(theme_dict)

count_theme_dict()
theme_dict = load_obj()
model = get_model()

correct, wrong = 0, 0

for i in get_data().values:
    words = done_text(str(i[1]))
    vec = 0
    for w in words:
        v = model.wv.get_vector(w)
        if vec == 0:
            vec = [v, 1]
        else:
            vec = [vec[0] + v, vec[1] + 1]
    vec = vec[0]/vec[1]

    min = -1
    id = 0
    for key, val in theme_dict.items():
        norm = np.linalg.norm(val - vec)
        if min == -1 or norm < min:
            min = norm
            id = key
    if id == i[0]:
        correct += 1
    else:
        wrong += 1
print(correct / (correct + wrong))
