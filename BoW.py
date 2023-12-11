# import library NLTK
import nltk
from sklearn.feature_extraction.text import CountVectorizer

class boW:
    x_train = []
    y_train = []
    #constructor
    def __init__(self, X_train):
        self.x_train = X_train

    #method
    def vectorizer(self):
        # create bag of words
        cv = CountVectorizer()
        X_train = cv.fit_transform(self.x_train).toarray()
        return X_train