import os
import numpy as np
import tensorflow as tf
from config import config
from util import DataUtils
from model import BasicRNN

# doing this only to suppress TF warning about softmax with logits being deprecated
tf.logging.set_verbosity(tf.logging.ERROR)

def get_data(dataUtil):
    '''
    Uses data utils to read data in textual form and convert it
    to vector form. Shuffles and returns two lists - one for sequences
    and another for labels in corresponding indices.
    '''
    # read in textual input
    x_data, y_true = dataUtil.get_sentences_labels()

    # create vectors at character level for each sequence and its label
    x_data, y_true = dataUtil.get_vectorized_data(x_data, y_true)

    # shuffle the above lists in parallel using random number generator
    rand_state = np.random.get_state()
    np.random.shuffle(x_data)
    np.random.set_state(rand_state)
    np.random.shuffle(y_true)

    return x_data, y_true

def main():
    '''
    Entry point for training and evaluating the model
    '''
    # initialize a data util object to read and process data from source directory
    dataUtil = DataUtils()

    # get sentences and labels in two lists in vector form
    x_data, y_true = get_data(dataUtil)

    # print dataset stats
    dataUtil.print_dataset_stats()

    # split dataset into train and test sets using sklearn
    x_train, y_train, x_test, y_test = dataUtil.custom_train_test_split(x_data, y_true)

    # get a basic RNN instance
    model = BasicRNN()

    # initialize the variables
    init = tf.global_variables_initializer()

    # create a tf saver object for saving and restoring models
    saver = tf.train.Saver()

    # training and evaluating
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(config.training.num_epochs):
            # run train step for the training data.
            # TODO: better to use batches here
            train_data = {model.X: x_train, model.y: y_train}
            sess.run(model.train_step, feed_dict=train_data)

            # print metrics every config.print_every steps
            if epoch % config.training.evaluate_frequency == 0:
                # calculate training accuracy
                train_accuracy = model.accuracy.eval(feed_dict=train_data)
                print("Step {0} : Training Accuracy = {1} %".format(epoch, train_accuracy))

                test_data = {model.X:x_test, model.y:y_test}
                # calculate testing accuracy
                test_accuracy = model.accuracy.eval(feed_dict=test_data)
                print("Evaluation Accuracy = {0} %".format(test_accuracy))

        # save model before the end of the session
        saver.save(sess, os.path.join(config.model.model_save_directory, config.model.model_name))
        print("Model saved in file: {0}".format(os.path.join(config.model.model_save_directory, config.model.model_name)))

if __name__== "__main__":
    main()
