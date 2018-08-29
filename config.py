import string

class DataConfig(object):
    # path to directory containing files. Note that each file has to contain sequences of
    # same sentiment and the name of the file must be the name of the sentiment
    source_directory = 'data/*'

    # including uppercase letter here
    all_valid_characters = string.ascii_letters + "0123456789-,;.!?:’\“/\\|_@#$%^&*~`+-=<>()[]{}' "
    num_valid_characters = len(all_valid_characters)

    # number and name of classes. this could ideally be generated from filenames
    # for a dataset provided there is one file for each sentiment.
    num_classes = 3
    all_classes = ['positive', 'neutral', 'negative']

    # max. sequence length should ideally be equal to length of longest sequence in the dataset
    # could be also set to 140 for a tweet in which case longer sequences are trimmed
    max_sequence_length = 186

class TrainingConfig(object):
    # percentage of data to use for train and test sets
    train_test_split = 0.8
    # number of steps after which model should be evaluated
    evaluate_frequency = 10
    # number of epochs
    num_epochs = 100

class ModelConfig(object):
    # location where model should be saved at the end of training
    model_name = 'model.ckpt'
    model_save_directory = 'tmp'

class Config(object):
    data = DataConfig()
    training = TrainingConfig()
    model = ModelConfig()

config = Config()