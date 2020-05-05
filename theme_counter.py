import pandas as pd
from preprocess_udpipe import done_text
import pickle
import scipy.spatial.distance as ds
import gensim.models
from gensim.corpora import Dictionary

THEME_DICT_FILENAME = 'obj/theme_dict.pkl'
REPORT_FILENAME = 'report.csv'
PROCESS_FILENAME = 'process.csv'
DIRECTION_FILENAME = 'direction.csv'


def get_data(dataset_filename):
    data = pd.read_csv(dataset_filename, encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    return data


def save_obj(obj):
    with open(THEME_DICT_FILENAME, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj():
    with open(THEME_DICT_FILENAME, 'rb') as f:
        return pickle.load(f)


def eval_tfidf(data):
    data1 = [done_text(str(d)) for d in data.text]
    dct = Dictionary(data1)
    corpus = [dct.doc2bow(line) for line in data1]
    tfidf = gensim.models.TfidfModel(corpus)
    dct.save('model_dict.dict')
    tfidf.save('tfidf_model.model')
    return dct, tfidf


def load_tfidf():
    dct = Dictionary.load('model_dict.dict')
    tfidf = gensim.models.TfidfModel.load('tfidf_model.model')
    return dct, tfidf


def count_theme_dict(model, dataset_filename):
    data = get_data(dataset_filename)
    dct, tfidf = eval_tfidf(data)
    theme_dict = {}
    for i in data.values:
        words = done_text(str(i[1]))
        for w in words:
            if w not in model.wv:
                continue
            vec = model.wv.get_vector(w)
            if not i[0] in theme_dict:
                theme_dict[i[0]] = [vec * tfidf.idfs[dct.token2id[w]], 1]
            else:
                theme_dict[i[0]] = [theme_dict[i[0]][0] + vec * tfidf.idfs[dct.token2id[w]], theme_dict[i[0]][1] + 1]

    for key, val in theme_dict.items():
        theme_dict[key] = val[0] / val[1]
    save_obj(theme_dict)


def test(model, dataset_filename):
    theme_dict = load_obj()

    correct, wrong = 0, 0
    ans = []
    j = 0

    dct, tfidf = load_tfidf()
    for i in get_data(dataset_filename).values:
        words = done_text(str(i[1]))
        vec = 0
        for w in words:
            if w not in model.wv:
                continue
            v = model.wv.get_vector(w)
            if vec == 0:
                vec = [v * tfidf.idfs[dct.token2id[w]], 1]
            else:
                vec = [vec[0] + v * tfidf.idfs[dct.token2id[w]], vec[1] + 1]
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
        ans.append([j, i[1], i[0], min, id, min1, id1, min2, id2])
        j += 1

        if id == i[0]:
            correct += 1
        else:
            wrong += 1

    save_test_ans(ans)
    return correct / (correct + wrong)


def save_test_ans(ans):
    process_df = pd.read_csv(PROCESS_FILENAME, encoding='utf-8', delimiter=';')
    process_dict = {}
    for i in process_df.values:
        process_dict[i[0]] = i[1]

    direction_df = pd.read_csv(DIRECTION_FILENAME, encoding='utf-8', delimiter=';')
    direction_dict = {}
    for i in direction_df.values:
        direction_dict[i[0]] = i[1]
    for val in ans:
        for i in [2, 4, 6, 8]:
            val[i] = process_dict[int(val[i][:5])] + "/" + direction_dict[int(val[i][6:])]
    ans.insert(0, [" ", "text", "real theme", "dist1", "theme1", "dist2", "theme2", "dist3", "theme3"])
    df = pd.DataFrame(ans)
    df.to_csv(REPORT_FILENAME, index=False, sep=";", encoding='utf-8')
