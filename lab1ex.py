##### EXERCISE 1##############

documents = [
    "this is an example sentence, nothing special happening in this sentence",
    "again a very ordinary sentence without anything special happening",
    "this is a sentence as example. this is another example sentence",
    "yet another sentence, simple and ordinary"
]


def tokenize(documents):
    tokenized = []
    for doc in documents:
        words = []
        for word in doc.split(''):
            if word[-1] in word.punctuation:
                words += [word[:-1, word[-1]]]
            else:
                words.append(word)
        tokenized.append(words)
    return tokenized


tokenized_docs = tokenize(documents)

from numpy.testing import assert_equal  # only have to import this once
from lab_1_solutions import tokenize as tokenize_solution

try:  # NOTE: make sure your function is in the same file / notebook
    assert_equal(tokenize(documents),
                 tokenize_solution(documents))
except AssertionError:
    print("Solution is not identical:")
    print("Your func output:", tokenize(documents))
    print("Solutions output:", tokenize_solution(documents))

##### EXERCISE 2##############
from collections import Counter


def vectorize(docs):
    T = sorted(set(word for doc in docs for word in doc))
    M = []
    for doc in docs:
        vector = []
        frequencies = Counter(doc)
        for t in T:
            vector = frequencies[t]
        M.append(vector)
    return M, T


term_frequency_matrix, T_dict = vectorize(tokenized_docs)

##### EXERCISE 3##############

import numpy as np


def cossim(M, T):
    sims = {}
    for i in range(len(M)):
        for j in range(i, len(M)):
            if i == j:
                continue
            p, q = M[i], M[j]
            sims[(i, j)] = np.dot(p, q) / (np.sqrt(np.dot(p, p)) *
                                           np.sqrt(np.dot(q, q)))
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[0][0]


similar = cossim(term_frequency_matrix, T_dict)

###### WITH LIBRARY SPACY ######

import spacy
nlp = spacy.load('en_core_web_sm')

sims = {}
for i in range(len(documents)):
    for j in range(i, len(documents)):
        if i == j:
            continue
        sims[(i, j)] = nlp(documents[i]).similarity(nlp(documents[j]))

print(sorted(sims.items(), key=lambda x: x[1], reverse=True)[0][0])

####### IN OOP STYLE WITH INHERITANCE #####

from collections import Counter
import string


class BaseVectorizer(object):

    def __init__(self):
        self.T = set()

    def tokenize(self, document):
        tokens = []
        for token in document.split(' '):
            if token[-1] in string.punctuation:
                tokens += [token[:-1], token[-1]]
            else:
                tokens.append(token)
        self.T.update(tokens)
        return tokens

    def get_vector(self, document):
        return NotImplemented

    def vectorize(self, documents):
        M = []
        for document in documents:
            M.append(self.get_vector(document))
        return M

    def fit(self, documents):
        return [self.tokenize(doc) for doc in documents]

    def transform(self, documents):
        assert self.T  # vocab not fitted
        return self.vectorize(documents)

    def fit_transform(self, documents):
        return self.transform(self.fit(documents))


class TFVectorizer(BaseVectorizer):

    def get_vector(self, document):
        vector = []
        freq_dict = Counter(document)
        for t in self.T:
            vector.append(freq_dict[t])
        return vector


class BinaryVectorizer(BaseVectorizer):

    def get_vector(self, document):
        vector = []
        for t in self.T:
            if t in document:
                vector.append(1)
            else:
                vector.append(0)
        return vector


vect = BinaryVectorizer()
print(vect.fit_transform(documents))

vect = TFVectorizer()
print(vect.fit_transform(documents))

####### IN OOP STYLE WITHOUT INHERITANCE #####

import numpy as np


class Similarity(object):

    def __init__(self, form='cosine'):
        self.sim_func = self.cossim if form == 'cosine' else self.eucsim

    def cossim(self, p, q):
        return np.dot(p, q) / (np.sqrt(np.dot(p, p)) *
                                       np.sqrt(np.dot(q, q)))
    def eucsim(self, p, q):
        return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

    def top_sims(self, M):
        sims = {}
        for i in range(len(M)):
            for j in range(i, len(M)):
                if i == j:
                    continue
                p, q = M[i], M[j]
                sims[(i, j)] = self.sim_func(p, q)
        return sorted(sims.items(), key=lambda x: x[1],
											reverse=True)[0][0]


vect = TFVectorizer()
M = vect.fit_transform(documents)
print(Similarity(form='cosine').top_sims(M))
print(Similarity(form='euclidean').top_sims(M))