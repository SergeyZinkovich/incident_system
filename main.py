import train
import theme_counter

DATASET_FILENAME = 'without_mess.csv'

print("=========WITHOUT UPDATE=========\n")
for size in range(10, 200, 10):
    train.train(DATASET_FILENAME, size, fasttext=True)
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

