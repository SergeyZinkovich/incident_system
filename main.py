import train
import theme_counter
import neural_network

DATASET_FILENAME = 'without_mess.csv'


def test_nn():
    w2v_model = train.load_trained_model()
    x, y, theme_dict, max_words = neural_network.prepare_data(DATASET_FILENAME, w2v_model)
    # x, y, theme_dict, max_words = neural_network.load_prepared_data()
    neural_network.train(w2v_model.vector_size, x, y, max_words, len(theme_dict))
    neural_network.test(x, y)


def test_w2v(with_update=False):
    if not with_update:
        print("=========WITHOUT UPDATE=========\n")
        for size in range(10, 200, 10):
            train.train(DATASET_FILENAME, size, fasttext=False)
            theme_counter.count_theme_dict(train.load_trained_model(), DATASET_FILENAME)
            print(size, " ")
            print(theme_counter.test(train.load_trained_model(), DATASET_FILENAME), "\n")
    else:
        print("=========WITH UPDATE=========\n")
        for size in range(10, 200, 10):
            train.train(DATASET_FILENAME, size)
            train.update(train.load_trained_model(), 'ruscorpora_mean_hs.model', True)
            theme_counter.count_theme_dict(train.load_updated_model(), DATASET_FILENAME)
            print(size, " ")
            print(theme_counter.test(train.load_updated_model(), DATASET_FILENAME), "\n")


test_nn()
# test_w2v()
