import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion

NGRAM_RANGE = (1, 4)
STOP_WORDS = None
LINEARSVC_LOSS = 'hinge'
LINEARSVC_C = 10**0.1


# Dummy preprocessor/tokenizer for ingredient counting
def itself(x):
    return x


# Processor to treat list of ingredients as one collection of words
# For ngram counting
def combine_words(ilist):
    return ' '.join(ilist)


# Functions to preprocess data.
def clean_ingr(ingr):
    SPEC_REMOVE = re.compile(r'(\'|\’|\(.*oz.*\)|(\()|(\)))')
    SPEC_AND = re.compile(r'\&')
    SPEC_ELSE = re.compile(r'[^\w\s\%_]')

    ingr = re.sub(SPEC_REMOVE, '', ingr)
    ingr = re.sub(SPEC_AND, 'and', ingr)
    ingr = re.sub(SPEC_ELSE, ' ', ingr)
    return ' '.join(ingr.split())


def get_ingrs(given):
    ingrs = [[clean_ingr(i).lower() for i in recipe['ingredients']]
             for recipe in given]
    return ingrs


def get_labels(given):
    return [r['cuisine'] for r in given]


class CookingModel():
    def __init__(self):
        self.vectorizer = FeatureUnion([
            ("ingrs", TfidfVectorizer(strip_accents='unicode',
                                      tokenizer=itself,
                                      preprocessor=itself)),
            ("words", TfidfVectorizer(strip_accents='unicode',
                                      preprocessor=combine_words,
                                      ngram_range=NGRAM_RANGE,
                                      stop_words=STOP_WORDS)),
            ])
        self.model = LinearSVC(loss=LINEARSVC_LOSS,
                               C=LINEARSVC_C)

    def fit(self, train, test=None):
        train_ingrs = get_ingrs(train)
        train_labels = get_labels(train)
        if test:
            test_ingrs = get_ingrs(test)
            self.vectorizer.fit(train_ingrs + test_ingrs)
        else:
            self.vectorizer.fit(train_ingrs)
        self.model.fit(self.vectorizer.transform(train_ingrs),
                       train_labels)
        return self.model

    def predict(self, data, to_kaggle=False):
        features = get_ingrs(data)
        preds = self.model.predict(self.vectorizer.transform(features))

        if to_kaggle:
            ids = [r['id'] for r in data]
            return pd.DataFrame({'id': ids, 'cuisine': preds},
                                columns=['id', 'cuisine'])
        else:
            return preds
