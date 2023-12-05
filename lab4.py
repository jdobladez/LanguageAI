'''
Last week, we have gone over the default setup for running 'vanilla' Machine Learning
algorithms using count-based features for classification (including tf*idf weighting),
using Scikit-learn. This week, we are going to move away from simple features, and into the
domain of more general representations of language: word embeddings. First, we need some comparison
material. Since (weighted) PPMI is fairly doable to implement (and not in Scikit-learn as-is), we'll
try to have a go at it. We can code it as a class with three different Counters:
'''

import numpy as np
from collections import Counter

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
            self.co_occ_freqs[context_word] += 1
        return self.co_occ_freqs
    def get_ppmi(self):
        return NotImplemented

'''
For the co-occurrence calculations as described in the Exercise, we need three things:
the total count of a word (word_freqs), the total count of a context (context_freqs),
and the co-occurrence count(co_occ_freqs). It's rather expensive to do summations over
column/rows in matrices every time, so keeping them in something like a hash map
(i.e., a dict) likely provides  faster look-up (disclaimer: haven't confirmed).
Either way, it's nicer to work with things we know than having to mess around with NumPy.
'''

'''
The annoying part of this is how to get the context windows in a not-super-convoluted fashion.
I will spoil a very neat one-liner for this (whaaat?  yes. — it's one of my favorite things in Python). Here it is:
'''

zip(*[tokens[i:] for i in range(context_window)])

# No way, right? Sure enough, if you run the following:

tokens = ["hello", "this", "is", "a", "sentence", "and", "that", "is", "all"]
print(list(zip(*[tokens[i:] for i in range(5)])))

'''
You will get all 5-grams in the text (🪄). It even returns an empty string if there are no grams of length.
 So how does it work? The slicing part is 'relatively' straight forward:
'''

print([tokens[i:] for i in range(5)])

'''
It generates $n$ (here 5) sequences, each of which has a 0 + $n$ starting position,
and then the rest of the sequence (:). If you try a very large $n$, you can see at some
point it runs out of sequence to take $n$th → until the end from. Ok, so what. We now
have a list of lists with potentially uneven sequences, but mostly of equal length. Where
does the magic happen? This is in the zip(* part. What happens here? zip is typically used
to concatenate two lists so that they are now a combination of tuples:
'''

a = [1, 2, 3]
b = [4, 5, 6]

for x in zip(a, b):
    print(x)
# or
for ai, bi in zip(a, b):
    print(ai, bi)

'''
Notice how it does element-wise looping over multiple lists at once?
What happens if we let it loop over two 1-off sequences of the same sentence?
'''
sentence = ["this", "is", "a", "sentence"]
part1 = ["this", "is", "a", "sentence"]  # or: sentence[0:]
part2 = ["is", "a", "sentence"]  # or: sentence[1:]

for x in zip(part1, part2):
    print(x)

'''
It gives bi-grams because of the element-wise iteration! If we add another sequence:
'''

sentence = ["this", "is", "a", "sentence"]
part1 = ["this", "is", "a", "sentence"]  # or: sentence[0:]
part2 = ["is", "a", "sentence"]  # or: sentence[1:]
part3 = ["a", "sentence"]  # or: sentence[2:]

for x in zip(part1, part2, part3):
    print(x)

'''
We get 3-grams, etc. But wait, they are uneven. It's not giving errors? Nope, zip stops iterating
if the smallest element runs out of things to loop through. So what's the asterisk doing there?
In our examples, we had separate lists to zip, but here we have a list of lists.
The * maps them to different 'positions' (as if they were individual variables).
A bit hacky, and not very interpretable, but still pretty amazing. Anyway, let's continue.
'''

'''
So for the last part, let's see if word embeddings improve performance on the task we have been
doing in all the previous lab sessions. For this, we need to turn it into a Vectorizer. gensim
used to provide this functionality, but not anymore. No worries, we can just make our own class:
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

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

'''
Here, we inherit some base functionality from scikit-learn that will provide the correct
class attributes to be compatible with their API (TransformerMixin for vectorizer classes,
BaseEstimator for standard objects). Note that during fit, we train word2vec on our data.
transform then converts them to vectors, and finally takes the mean over the vector dimensions (axis=1)
to return it as one vector representation. We might then for example train a Multi-Layer Perceptron 
(Feedforward Network) on these embeddings:
'''

from sklearn.neural_network import MLPClassifier
w2v = Word2VecTransformer()
X = w2v.fit_transform([t['text'] for t in dataset])
# split the data into train and test
# NOTE: we don't have labels, this is just some pseudo-code example
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)