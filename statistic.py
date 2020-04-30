import train
import theme_counter
from preprocess_udpipe import done_text
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def words_for_themes():
    theme_dict = theme_counter.load_obj()
    model = train.load_trained_model()
    for theme in theme_dict.values():
        print(model.wv.most_similar(positive=[theme], topn=1))


def words_for_texts():
    texts = theme_counter.get_data('file.csv').values
    model = train.load_trained_model()
    for text in texts:
        words = done_text(str(text[1]))
        vec = 0
        for w in words:
            v = model.wv.get_vector(w)
            if vec == 0:
                vec = [v, 1]
            else:
                vec = [vec[0] + v, vec[1] + 1]
        vec = vec[0] / vec[1]
        print(model.wv.most_similar(positive=[vec], topn=1))


def words_plot():
    model = train.load_trained_model()
    X = model.wv[model.wv.vocab]
    X_tsne = TSNE(n_components=2).fit_transform(X)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()


def themes_plot():
    theme_dict = theme_counter.load_obj()
    X = [i for i in theme_dict.values()]
    X_tsne = TSNE(n_components=2).fit_transform(X)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.show()


themes_plot()