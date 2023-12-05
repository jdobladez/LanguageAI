import numpy as np
def count(x):
    c = np.bincount(x)
    return c[c.nonzero()]

def entropy(x, y):
    return -(sum(x / y * np.log2(x / y)))

def information_gain(x, y):
    IG = entropy(count(y), len(y))
    for value in np.unique(x):
        y_split = y[np.where(x == value)]
        E = entropy(count(y_split), len(y_split))
        IG -= (len(y_split) / len(y)) * E
    return IG

def best_split_feature(X, y):
    scores = [(i, information_gain(x, y)) for i, x in enumerate(X.T)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]