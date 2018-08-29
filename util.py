import os
import glob
import numpy as np
import unicodedata
from io import open
from config import config
from sklearn.model_selection import train_test_split

class DataUtils(object):

    def __init__(self):
        '''
        Initializes data structures to store data read from given files.
        Assumes no prior knowledge of type and num. of classes.
        '''
        self.labels = []
        self.sequences = []
        self.label_sequence_dict = {}
        self.unknown_characters = set()
        self.num_classes = config.data.num_classes
        self.all_classes = config.data.all_classes

    def find_files(self, path):
        '''
        Returns a list of paths for files found in given path.
        '''
        return glob.glob(path)

    def convert_unicode_to_ascii(self, sequence):
        '''
        Converts a Unicode string to plain ASCII
        '''
        return ''.join(
            char for char in unicodedata.normalize('NFD', sequence)
            if unicodedata.category(char) != 'Mn'
            and char in config.data.all_valid_characters
        )

    def get_lines_from_file(self, filename):
        '''
        Opens a file, returns all the lines read in their ASCII form
        '''
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.convert_unicode_to_ascii(line) for line in lines]

    def get_sentences_labels(self):
        '''
        Returns a list of sequences and labels in the order they are read from
        files present in the source directory.
        '''
        for filename in self.find_files(config.data.source_directory):
            class_name = os.path.basename(filename)
            lines = self.get_lines_from_file(filename)
            self.label_sequence_dict[class_name] = lines
            for line in lines:
                self.sequences.append(line)
                self.labels.append(class_name)
        config.data.max_sequence_length = len(max(self.sequences, key=len))
        return self.sequences, self.labels

    def print_dataset_stats(self):
        '''
        Prints some stats for the dataset provided under source directory.
        '''
        print('-------------')
        print('DATASET STATS')
        print('All Classes : ', self.all_classes)
        print('Classwise Distribution')
        for label, sequences in self.label_sequence_dict.items():
            print('{0} - {1}'.format(label, len(sequences)))
        print('Total Sequences : ', len(self.sequences))
        print('Total Corresponding Labels : ', len(self.labels))
        print('Longest sentence found : ', max(self.sequences, key=len))
        print('Length of longest sentence : ', len(max(self.sequences, key=len)))
        print('Unknown characters found : ', self.unknown_characters)
        print('-------------')

    def get_sequence_vector(self, sequence):
        '''
        Creates a numpy array for a string where each character within the string is
        one hot encoded. Returns a [len(sequence)*num_valid_characters] size np array.
        '''
        seq2vec = []
        for char in sequence:
            char2vec = np.zeros(config.data.num_valid_characters, dtype=np.int)
            if(char in config.data.all_valid_characters):
                char2vec[config.data.all_valid_characters.index(char)] = 1
            else:
                self.unknown_characters.add(char)
            seq2vec.append(char2vec)
        while len(seq2vec) < config.data.max_sequence_length:
            seq2vec.append(np.zeros(config.data.num_valid_characters, dtype=np.int))
        seq2vec = np.array(seq2vec)
        return seq2vec

    def get_label_vector(self, label):
        '''
        Returns a 1*num_labels np array for a given label
        '''
        label2vec = np.zeros(self.num_classes, dtype=np.int)
        label2vec[self.all_classes.index(label)] = 1
        return label2vec

    def get_vectorized_data(self, sequences, labels):
        '''
        Uses the above defined utility functions to convert a list of strings
        to their vector representations.
        '''
        sequence_vectors = []
        label_vectors = []
        for sequence in sequences:
            sequence_vectors.append(self.get_sequence_vector(sequence))
        for label in labels:
            label_vectors.append(self.get_label_vector(label))
        return sequence_vectors, label_vectors

    def custom_train_test_split(self, x_data, y_true):
        '''
        Splits given lists into train and test sets based on the split
        ratio defined in the config using sklearn and returns np arrays
        '''
        split_ratio = config.training.train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_true, test_size=split_ratio)
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
