import pandas as pd
from preprocess_udpipe import done_text
import numpy as np
import pickle
import scipy.spatial.distance as ds

THEME_DICT_FILENAME = 'obj/theme_dict.pkl'
REPORT_FILENAME = 'report.csv'


def get_data(dataset_filename):
    data = pd.read_csv(dataset_filename, encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    return data


def save_obj(obj):
    with open(THEME_DICT_FILENAME, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj():
    with open(THEME_DICT_FILENAME, 'rb') as f:
        return pickle.load(f)


def count_theme_dict(model, dataset_filename):
    theme_dict = {}
    for i in get_data(dataset_filename).values:
        words = done_text(str(i[1]))
        for w in words:
            vec = model.wv.get_vector(w)
            if not i[0] in theme_dict:
                theme_dict[i[0]] = [vec, 1]
            else:
                theme_dict[i[0]] = [theme_dict[i[0]][0] + vec, theme_dict[i[0]][1] + 1]

    for key, val in theme_dict.items():
        theme_dict[key] = val[0] / val[1]
    save_obj(theme_dict)


def test(model, dataset_filename):
    theme_dict = load_obj()

    correct, wrong = 0, 0

    ans_csv = {"Theme": ["1", "2", "3", "4", "5", "6", "7"]}

    j = 0

    for i in get_data(dataset_filename).values:
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
        min1 = -1
        min2 = -1
        id = 0
        id1 = 0
        id2 = 0
        for key, val in theme_dict.items():
            norm = ds.cosine(val, vec)
            if min == -1 or norm < min:
                min2 = min1
                id2 = id1
                min1 = min
                id1 = id
                min = norm
                id = key
            else:
                if min1 == -1 or norm < min1:
                    min2 = min1
                    id2 = id1
                    min1 = norm
                    id1 = key
                elif min2 == -1 or norm < min2:
                    min2 = norm
                    id2 = key
        ans_csv[j] = [i[0], min, id, min1, id1, min2, id2]
        j += 1

        if id == i[0]:
            correct += 1
        else:
            wrong += 1

    df = pd.DataFrame(ans_csv)
    df.to_csv(REPORT_FILENAME, index=True, sep=";")
    return correct / (correct + wrong)

