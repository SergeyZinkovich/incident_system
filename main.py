import train
import theme_counter
import neural_network

DATASET_FILENAME = 'without_mess.csv'


def test_nn():
    x, y, theme_dict, max_words, unic_words_count = neural_network.prepare_data(DATASET_FILENAME)
    # x, y, theme_dict, max_words, unic_words_count = neural_network.load_prepared_data()
    neural_network.train(x, y, max_words, len(theme_dict), unic_words_count)
    neural_network.test(x, y)


def test_w2v():
    print("=========WITHOUT UPDATE=========\n")
    for size in range(10, 200, 10):
        train.train(DATASET_FILENAME, size, fasttext=False)
        theme_counter.count_theme_dict(train.load_trained_model(), DATASET_FILENAME)
        print(size, " ")
        print(theme_counter.test(train.load_trained_model(), DATASET_FILENAME), "\n")

    print("=========WITH UPDATE=========\n")
    for size in range(10, 200, 10):
        train.train(DATASET_FILENAME, size)
        train.update(train.load_trained_model(), 'ruscorpora_mean_hs.model', True)
        theme_counter.count_theme_dict(train.load_updated_model(), DATASET_FILENAME)
        print(size, " ")
        print(theme_counter.test(train.load_updated_model(), DATASET_FILENAME), "\n")


test_nn()
# test_w2v()
