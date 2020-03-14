import pandas as pd
import gensim.downloader as api
import gensim
from gensim.test.utils import common_texts
from gensim.sklearn_api import W2VTransformer
import gensim.models
from gensim.utils import simple_preprocess
from preprocess_udpipe import done_text

def get_data():
    data = pd.read_csv('file.csv', encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    data = [str(d) for d in data.text]
    return map(done_text, data)


def train():
    sentences = get_data()

    model = gensim.models.Word2Vec(sentences=sentences, min_count=0, size=130) #130 = 43
    model.save('models/my_model.model')

def load():
    return gensim.models.Word2Vec.load('models/my_model.model')

def update(model):
    sentences = get_data()
    wv = gensim.models.KeyedVectors.load('models/araneum_none_fasttextcbow_300_5_2018.model')
    model.build_vocab(list(wv.vocab.keys()), update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.save('models/work_model.model')

    #print(model.wv.most_similar('коректировки'))



train()
model = load()
#update(model)








