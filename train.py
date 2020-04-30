import pandas as pd
import gensim
import gensim.models
from preprocess_udpipe import done_text
import zipfile

TRAINED_MODEL_FILENAME = 'models/trained_model.model'
UPDATED_MODEL_FILENAME = 'models/updated_model.model'


def get_data(dataset_filename):
    data = pd.read_csv(dataset_filename, encoding='utf-8', delimiter=';', header=None, names=['id', 'text'])
    data = [str(d) for d in data.text]
    return map(done_text, data)


def train(dataset_filename, size=100, fasttext=False):
    sentences = get_data(dataset_filename)
    if fasttext:
        model = gensim.models.FastText(sentences=sentences, min_count=0, size=size)
    else:
        model = gensim.models.Word2Vec(sentences=sentences, min_count=0, size=size)
    model.save(TRAINED_MODEL_FILENAME)


def load_madel(model_filename):
    return gensim.models.Word2Vec.load(model_filename)


def load_trained_model():
    return load_madel(TRAINED_MODEL_FILENAME)


def load_updated_model():
    return load_madel(UPDATED_MODEL_FILENAME)


def update(model, pretrained_model_name, binary=False):
    if binary:
        with zipfile.ZipFile('models/' + pretrained_model_name + '.zip', 'r') as archive:
            stream = archive.open(pretrained_model_name + '.bin')
            wv = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    else:
        wv = gensim.models.KeyedVectors.load('models/' + pretrained_model_name + '.model')
    model.build_vocab(list(wv.vocab.keys()), update=True)
    model.train(list(wv.vocab.keys()), total_examples=model.corpus_count, epochs=model.iter)
    model.save(UPDATED_MODEL_FILENAME)
