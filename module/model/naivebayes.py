import numpy as np
from sklearn.naive_bayes import MultinomialNB

#[A, B C, D] => cls1: (A), (B, C, D), cls2: (B), (A, C, D), ...


class NaiveBayes(object):
    def __init__(self, classes):
        self.model = {}
        self.classes = classes

        for cls in self.classes:
            model = MultinomialNB()
            self.model[cls] = model

    def fit(self, train_x, train_y):
        for idx, cls in enumerate(self.classes):
            class_labels = train_y[:, idx]
            self.model[cls].fit(train_x, class_labels)


    def predict(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            predictions[:, idx] = self.model[cls].predict(test_x)
        return predictions
    
    def predict_prob(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            predictions[:, idx] = self.model[cls].predict_proba(test_x)[:, 1]
        return predictions