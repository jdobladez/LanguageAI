'''
So we've worked on representing text as vectors in Practical 1, and we've prepared the data (see Practical 2),
and got some intuitions about the task (classifying Subreddits by topic). Before we put things together,
let's work on implementing a classifier we have seen in the Lecture—from scratch.
We'll try the ID3 Decision Tree, and take the example from the Exercises this week to test:
'''
import numpy as np

X = np.array([
    [1, 2, 0, 1, 2, 1]
]).T  # transpose to make this a feature column
y = np.array([0, 1, 1, 0, 0, 1])

'''
INFORMATION GAIN (TOPIC)
Let's start out with calculating Information Gain for a particular feature first, as that is basically our
recursive operation to determine the left/right/etc. splits of the tree. Above, I formatted the single
feature as a matrix so that we can make the function more general as we go. Below, you can see a minimal
implementation of the parts around calculating Information Gain. Let's break this down.
'''

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

information_gain(X[:,0], y)

'''
So we need two things: a count per value of $S$, and the total $S$. The latter we can simply get with len,
but for the former we need a count function (NotImplemented). Once we have those, we can submit them to entropy,
which should sum the product of $p_i$ and $\log_2\ p_i$. You can do this using array operations, or in a loop.
Once we have this functionality down, we can basically repeat the same steps for every $v \in x_n$; i.e.,
we need a summation loop for several entropy values. What is left, then, is taking 'splits' from the data;
we want to know the counts for a particular $v$ split by label (to do $|Y_v|\ /\ |Y|$, and Entropy).
This is handled by this particular line:
'''

xy_split = y[np.where(x == value)]

'''
This indexes into $Y$ (y[...]), and takes only the values where $x_n = v$.
To test what this does, you can try the following:
'''
x = X[:,0]  # our feature x_n
v = 1       # the feature value 1
print(x)
print(np.where(x == v))  # these are indices of x
print(y[np.where(x == v)])

# The IG part requires yv / y and entropy to be implemented still.

'''
So, we can calculate IG per feature (using information_gain), which means we can rank our features,
determine which one we make our first split on, and make the split based on the values. We need an outer
loop over X.T (to loop per feature), store, and sort the IGs, and return the column index of the feature
(best_split_feature). The code for handling the splits once the best index is known (NotImplemented)
is given below (split_data): 
'''

def best_split_feature(X, y):
    return NotImplemented

def split_data(X, y):
    top_ix = best_split_feature(X, y)
    for v, split_ix in [(v, np.where(X[i, :] == v))
                        for v in np.unique(X[i,:])]:
        yield top_ix, v, X[split_ix], y[split_ix]

# You can use this matrix for this part:

X = np.array([
    [1, 0, 0, 1, 2],
    [2, 0, 1, 1, 0],
    [1, 1, 0, 0, 2],
    [1, 2, 0, 0, 1],
    [1, 0, 0, 1, 2],
    [0, 0, 0, 1, 0]
])
y = np.array([0, 1, 1, 0, 0, 1])

'''
PAINLESS PREDICTION (SKLEARN)
Scikit-learn (sklearn) is a fantastic library to get started with several prediction problems, 
including text classification. We'll focus on 'the works', but—as I assume the majority of you are
familiar with the library—mostly on integrating spaCy, Pipelines, and (fair) evaluation.
This part will mostly be library-driven, but I expect you to be familiar with the parts explained here,
and to be able to apply them to new problems.
'''

'''
GETTING VECTORS FOR DATA
Remember that in the previous Lab Session, we loaded text data from political Subreddits, using Pandas, like so:
'''
import pandas as pd

# mirror https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download
df = pd.read_csv('https://onyx.uvt.nl/grabber/politics/reddit-5000.csv',
                 na_values=['[deleted]', '[removed]']).dropna()
df.head(5)

'''
Scikit-learn has a compatibility layer for pandas, so we can get to work immediately. Let's start
by splitting our data into a train and test set, using train_test_split (see docs).
Note that our dataset is not nicely distributed:
'''

import matplotlib.pyplot as plt
ax = df['subreddit'].value_counts().plot(kind='bar')
plt.show()

'''
We have a few options:
1. We **undersample** the majority class—randomly being the most naive approach. 
More elegant approaches generally rely on [distance]: we'd probably first want to exclude instances
that are very similar, rather than throwing potentially informative ones away at random.
Generally this a good option if the other data is plentiful.
2. We **oversample** the minority class (see e.g. [SMOTE]) by generating synthetic data.
This is a really challenging problem for text data, so I would tread carefully.
3. We account for the class imbalance using **stratified sampling**, and 
class weights in classifiers (if they support this). 
4. We ignore this and hope for the best (wouldn't recommend this personally).

For this practical, we'll stratify and hope for the best. Once we have these sets, we can vectorize them
 Scikit-learn offers a few different flavors. Note that these have different parameter settings, one of which
  is the ability to pass a [custom tokenizer].
'''

'''
Here we'll test the very useful Pipeline environment; it allows us to define several steps, 
and as all have `fit` and `transform` methods, they chain easily under one class. You can see 
it in action, but I'll walk through setting this up for our current ask.

Before we defined tf*idf, but we might want to tune some options there as well,
such as lemmatization instead of simple tokenization, $n$-gram ranges, case folding yes 
or no, you name it. We'll therefore also incorporate it as an element in our hyperparameter tuning.

As classifier, let's stick to Multinomial Naive Bayes for now; it tends to work well with tf*idf, 
and does not require any tuning. This whole lot (if you distill it from the page), looks like so:
'''

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

'''
So note that our pipeline is a list of tuples, where the first element is a str with a name identifier.
You can also set hyperparameters you'd like to keep static in this step. Then, for the parameters, 
we use a dict where the key is "str name id + __ + the parameter name" (e.g., tfidf__tokenizer),
and then a list of parameters as value. You can see the tokenizer example in the code above.
We then run Grid Search in a cross-validation setting (docs). It takes these two elements, and 
has addition arguments. It is fitted similar to the classifiers/vectorizers. Via the class methods,
we can extract the parameters for the best performing classifier in the grid. Finally, 
we print these best-performing parameters.
'''