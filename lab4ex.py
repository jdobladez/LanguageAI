################ EXERCISE 1 ##############
import numpy as np
from collections import Counter

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
X = fetch_20newsgroups(subset='train', categories=['sci.space']).data

class PPMIVectorizer(object):

    def __init__(self, alpha=0.75, context_window=5):
        self.word_freqs = Counter()
        self.context_freqs = Counter()
        self.co_occ_freqs = Counter()
        self.alpha = alpha
        self.context_window = context_window

    def get_counts(self, document):
        for window in zip(*[document[i:] for i in range(self.context_window)]):
            middle = int((len(window) - 1) / 2)
            context = window[:middle] + window[middle + 1:]
            word = window[middle]
            self.word_freqs[word] += 1
            for context_word in context:
                self.context_freqs[context_word] += 1
                self.co_occ_freqs[(word, context_word)] += 1
        return self.co_occ_freqs

    def get_ppmi(self):
        sum_context = sum(self.context_freqs.values())
        sum_word = sum(self.word_freqs.values()) + sum_context
        for (w, c), wc_freq in self.co_occ_freqs.items():
            p_wc = wc_freq / sum_word
            p_w = self.word_freqs[w] / sum_word
            p_alpha_c = (self.context_freqs[c] ** self.alpha /
                         sum_context ** self.alpha)
            ppmi = max(np.log2(p_wc/(p_w*p_alpha_c)), 0)
            self.co_occ_freqs[(w, c)] = ppmi
        return self.co_occ_freqs


# TEST FUNCTION
from numpy.testing import assert_equal  # only have to import this once
from lab_4_solutions import PPMIVectorizer as PPMIVectorizer_solution

try:  # NOTE: make sure your class is in the same file / notebook
    assert_equal(PPMIVectorizer().get_counts(X).most_common(20),
                 PPMIVectorizer_solution().get_counts(X).most_common(20))
    print("Success!")
except AssertionError:
    print("Solution is not identical:")
    print("Your func output:",
          PPMIVectorizer().get_counts(X).most_common(20))
    print("Solutions output:",
          PPMIVectorizer_solution().get_counts(X).most_common(20))

################ EXERCISE 2 ##############
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

dataset = api.load("fake-news")

model = Word2Vec([simple_preprocess(t['text']) for t in dataset])

sims = [{w: sim for w, sim in model.wv.most_similar(word, topn=100)}
        for word in ["trump", "biden"]]

df = pd.DataFrame(sims)
df.dropna(axis=1, inplace=True)
print(df)

'''
similarity weights change per run! Word2vec not deterministic (so not reproducible) out of the box. This can be
fixed by providing the function with a custom hash (not a great implementation here) and setting
workers to 1 (multi-threading messes up single-program-determinism):
'''
def pseudo_hash(astring):
    return ord(astring[0])

dataset = api.load("fake-news")
model = Word2Vec([simple_preprocess(t['text']) for t in dataset], workers=1, hashfxn=pseudo_hash)

sims = [{w: sim for w, sim in model.wv.most_similar(word, topn=100)}
        for word in ["trump", "biden"]]

df1 = pd.DataFrame(sims)
df1.dropna(axis=1, inplace=True)
print(df1)

################ EXERCISE 3 ##############

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download',
                 na_values=['[deleted]', '[removed]']).dropna()

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['subreddit'], test_size=0.1, random_state=42,
                                                    stratify=df['subreddit'])

class Word2VecTransformer(TransformerMixin,BaseEstimator):
    def __init__(self):
        self.w2v = None

    def fit(self, X, y=None):
        self.w2v = Word2Vec([simple_preprocess(doc) for doc in X])
        return self

    def transform(self, X):
        vec_X = []
        for doc in X:
            vec = []
            for token in simple_preprocess(doc):
                try:
                    vec.append(self.w2v.wv[token])
                except KeyError:
                    pass
            if not vec:
                vec.append(self.w2v.wv['the'])
            vec_X.append(np.mean(vec, axis=1))
        return vec_X

''' GETTING AN ERROR FOR EX3
tok = Word2VecTransformer()
tfidf = TfidfVectorizer(sublinear_tf=True, tokenizer=tok.transform)
X_tr_tf = tfidf.fit_transform(X_train)
X_te_tf = tfidf.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("tfidf", Word2VecTransformer()),
    ("clf", MultinomialNB()),
])

parameters = {
    "tfidf__ngram_range": ((1, 1), (1, 2)),  # unigrams or uni+bigrams
    "tfidf__lowercase": [True, False],       # lowercase yes / no
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()

print("Best parameter settings:")
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

model = grid_search.best_estimator_

from sklearn.metrics import classification_report
ŷ = model.predict(X_test)  # ŷ can also be y_pred for older versions
print(classification_report(y_test, ŷ))

ŷ = ['Conservative'] * len(y_test)
print(classification_report(y_test, ŷ))

'''