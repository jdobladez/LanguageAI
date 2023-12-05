################# EXERCISE 1 ############################
import numpy as np

def count(x):
    count = np.bincount(x)
    return count[count.nonzero()]

def entropy(x, y):
    return -(sum((x/y)*np.log2(x/y)))

def information_gain(x, y):
    IG = entropy(count(y), len(y))
    for value in np.unique(x):
        y_split = y[np.where(x == value)]
        E = entropy(count(y_split), len(y_split))
        IG -=  E*(len(y_split)/len(y))# implement
    return IG

################# EXERCISE 2 ############################


def best_split_feature(X, y):
    scores = [(i, information_gain(x, y)) for i, x in enumerate(X.T)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]

X = np.array([
    [1, 0, 0, 1, 2],
    [2, 0, 1, 1, 0],
    [1, 1, 0, 0, 2],
    [1, 2, 0, 0, 1],
    [1, 0, 0, 1, 2],
    [0, 0, 0, 1, 0]
])
y = np.array([0, 1, 1, 0, 0, 1])

################# EXERCISE 3 ############################

from sklearn.model_selection import train_test_split

import pandas as pd

# mirror
df = pd.read_csv('https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download',
                 na_values=['[deleted]', '[removed]']).dropna()
df.head(5)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['subreddit'], test_size=0.1, random_state=42,
                                                    stratify=df['subreddit'])
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

class Tokenizer(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize(self, doc):
        return [token.text for token in self.nlp(doc)]

    def lemmatize(self, doc):
        return [token.lemma_ for token in self.nlp(doc)]

tok = Tokenizer()
tfidf = TfidfVectorizer(sublinear_tf=True, tokenizer=tok.tokenize)
X_tr_tf = tfidf.fit_transform(X_train)
X_te_tf = tfidf.transform(X_test)

################# EXERCISE 4 ############################


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True)),
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