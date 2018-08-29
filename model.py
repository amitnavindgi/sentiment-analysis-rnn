import numpy as np
import tensorflow as tf
from config import config

class BasicRNN(object):

    def __init__(self):
        '''
        Initializes parameters and creates placeholders for input data
        '''
        # training parameters
        self.learning_rate =  0.005
        # num. of hidden units
        self.num_hidden = 128
        # num of classes
        # self.num_classes = dataUtil.num_classes
        self.num_classes = config.data.num_classes

        # create placeholders
        self.X = tf.placeholder(tf.float32, [None, config.data.max_sequence_length, config.data.num_valid_characters], name='input_x')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')

        # cell
        self.cell = tf.contrib.rnn.BasicRNNCell(self.num_hidden)

        # use dynamic rnn and grab outputs
        self.outputs, self.states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)

        # set weight and biases
        self.W = tf.Variable(tf.truncated_normal([self.num_hidden, self.num_classes],stddev=0.1), name='weight')
        self.b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='bias')

        # predict y based on final RNN output
        self.y_hat = tf.add(tf.matmul(self.outputs[:,-1,:], self.W), self.b, name='prediction')

        # compute the cross entropy loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_hat, labels=self.y))

        # create an optimizer and define training operation where loss needs to be minimized
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        self.train_step = self.optimizer.minimize(self.loss)

        # compute metrics
        self.predicted_label = tf.argmax(self.y_hat,1, name='predicted_label')
        self.correct_label = tf.argmax(self.y,1)
        self.correct_prediction = tf.equal(self.predicted_label, self.correct_label)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')
        self.accuracy = tf.multiply(100.0, self.accuracy, name='accuracy_percentage')
