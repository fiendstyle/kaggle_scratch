import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv

class Preprocessor(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = self.config['classes']
        self._load_data()

    @staticmethod
    def clean_text(text):
        text = text.strip().lower().replace('\n', '')
        words = re.split(r'\W+', text)
        filter_table = str.maketrans('', '', string.punctuation)
        clean_words = [w.translate(filter_table) for w in words if len(w.translate(filter_table))]
        return clean_words
    
    def _parse(self, data_frame, is_test = False):
        """

            parameters:
                data_frame
            returns:
                tokenized_input(np.array): # [i, havent, t, xx]
                n_hot_lable(np.array):     # [0, 0, 0, 0, 0, 1]
                        or
                test_ids if is_test = True
        """
        x = data_frame[self.config['input_text_column']].apply(Preprocessor.clean_text).values
        y = None
        if not is_test:
            y = data_frame.drop([self.config['input_id_column'], self.config['input_text_column']], axis=1).values
        else:
            y = data_frame.id.values
        return x, y

    def _load_data(self):
        data_df = pd.read_csv(self.config['input_trainset'])
        self.logger.info("train lines: {}".format(data_df.size))
        data_df.fillna({self.config['input_text_column'] : "unknow"}, inplace=True)
        self.data_x, self.data_y = self._parse(data_df)
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(
            self.data_x, self.data_y,
            test_size = self.config['split_ratio'],
            random_state = self.config['random_seed']
        )
        test_df = pd.read_csv(self.config['input_testset'])
        self.logger.info("test lines: {}".format(test_df.size))
        test_df.fillna({self.config['input_text_column'] : "unknow"}, inplace=True)
        self.test_x, self.test_ids = self._parse(test_df, is_test=True)

    def process(self):
        input_convertor = self.config.get('input_convertor', None)

        data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = \
            self.data_x, self.data_y, self.train_x, self.train_y, \
            self.validate_x, self.validate_y, self.test_x
        
        if input_convertor == 'count_vectorization':
            train_x, validate_x = self.count_vectorization(train_x, validate_x)
            data_x, test_x = self.count_vectorization(data_x, test_x)
        elif input_convertor == 'tfidf_vectorization':
            train_x, validate_x = self.tfidf_vectorization(train_x, validate_x)
            data_x, test_x = self.tfidf_vectorization(data_x, test_x)

        return data_x, data_y, train_x, train_y, validate_x, validate_y, test_x
    
    def count_vectorization(self, train_x, test_x):
        vectorizer = CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_test_x = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_test_x
    
    def tfidf_vectorization(self, train_x, test_x):
        vectorizer = TfidfVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_test_x = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_test_x