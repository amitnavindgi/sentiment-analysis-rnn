import os
import argparse
import numpy as np
import tensorflow as tf
from config import config
from util import DataUtils
from model import BasicRNN

# doing this only to suppress TF warning about softmax with logits being deprecated
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predicts sentiment for user given text.')
    parser.add_argument('--text',
                        nargs='+',
                        required=True,
                        help='Text in quotes for which sentiment needs to be predicted')
    args = parser.parse_args()

    if args.text:
        # args will be list, join them
        sequence = ' '.join(args.text)

        # exit if len of user input is more than defined embedding size.
        # if it is longer it can be trimmed too but not doing it here.
        if len(sequence) > config.data.max_sequence_length or len(sequence) == 0:
                print('Invalid input. Max. text length is 140')
                exit()

        dataUtil = DataUtils()
        graph = tf.Graph()

        with graph.as_default():
            with tf.Session() as sess:
                # restore the meta graphs and the model
                saver = tf.train.import_meta_graph(os.path.join(config.model.model_save_directory, config.model.model_name) + '.meta')
                saver.restore(sess, tf.train.latest_checkpoint(config.model.model_save_directory))

                # verify if weights are being restored by expecting same values
                # when run twice
                # weight = sess.run(graph.get_tensor_by_name('weight:0'))
                # print(weight[:5][:5])
                input_x = graph.get_tensor_by_name('input_x:0')

                # create char level embedding for user given sequence
                seq2vec = dataUtil.get_sequence_vector(sequence)
                x_data = np.expand_dims(seq2vec, axis=0)

                # restore output operation
                y_hat = graph.get_operation_by_name('prediction').outputs[0]

                # prepare feed data and calculate output
                input_data = {input_x: x_data}
                prediction = sess.run(y_hat, feed_dict=input_data)

                # print model prediction to stdout
                # print(prediction) -> use this to see if the output doesnt make sense ;)
                # argmax choose the first index if there are two max values
                # print(np.argmax(prediction))
                print(config.data.all_classes[np.argmax(prediction)])
