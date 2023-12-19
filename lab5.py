"""
HOW TO WRITE CODE IN A NON-MESSY WAY:

Sections (Classes) that should be present:

1. Dataloader
2. Descriptives: reads a data state and reports some corpus statistics
3. Modeler: provided with a set of (baseline) models, hyper-parameters, and data. Fits model, predicts, and outputs
4. Logger: (optional) wraps around modeler, keeps track of experimental state
5. Evaluator: using prediction files, runs eval metrics and qualitative evaluations
6. Tester: (optional) write nose tests that run dummy data/models through pipeline. It should test assumptions 
    via assert statements
7. Experiment: serves as configuration class that calls all elements above in correct order and passes user settings
    by settings.yml or settings.json or argparse

Ideally, each class is in separate files.

Use small # comments, merge them into docstrings (huge comment section) later, If variables names are obvious, then most
10-20 line functions should be readable without comment per line.

Keep a .md file  to write high-level documentation while writing code.
"""

# This is an example of very messy code snippets:

import matplotlib

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset="train")
y, X = data.target, data.data
y, X = zip(*[(yi, xi) for yi, xi in zip(y, X)
           if yi in [2, 4, 3]])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

tfidf = CountVectorizer()
X_tf = tfidf.fit_transform(X)
nb = MultinomialNB()
nb.fit(X_tf.todense() * 1000, y)

feature_names = tfidf.get_feature_names()
coefs_with_fns = sorted(zip(nb.feature_log_prob_[0], feature_names))
top = zip(coefs_with_fns[:50], coefs_with_fns[:-(50 + 1):-1])
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

coefs_with_fns[:10]

coefs_with_fns[-10:]

coefs_with_fns = sorted(zip(nb.feature_log_prob_[1], feature_names))
coefs_with_fns[-10:]

import matplotlib.pyplot as plt

values, features = zip(*coefs_with_fns[50000:50050])
plt.figure(figsize=(15, 2))
plt.bar(features, [-1 * x for x in values])


plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Coefficients')
plt.xticks(rotation=45)

plt.show()

"""
HOW TO WRITE A README FILE FOR AN ACADEMIC PROJECT:

Example of a README file: https://github.com/cmry/reap

The page should have a rough description of information conveyed like:

1. Header with vital info: what/who/where/why is answered quickly here, via a link to the paper, distinct 
    software license(s), data (if provided), and .bib file to cite work
2. Section with useful points for reproduction: tl;dr (highlights points of importance), instructions on how to
    reproduce the results and data (if provided), and what system it was built on (Python version & OS).
3. Section dedicated to experimental manipulation: what elements can be changed to change experiment? where to do
    these changes? State this by adding the line numbers and the file corresponding
4. Section dedicated to how to add to the research code: which components are modular and can be swapped out? How does 
    one do that? 

HOW TO STRUCTURE THE REPOSITORY OR CODE:

In this paper: https://github.com/cmry/simple-queries

The different functions and classes are structured according to their section in the paper. Each file is dedicated
    to a main thing, for example: misc_keys.py is where the Twitter API is, and sec3_data.py is where the data is 
    collected.

In this paper: https://github.com/cmry/amica

A factory class was used; calls other classes with required parameters and in the right order.
The file evaluation.py has the Evaluation Class which handles cross-validation, tuning, reporting on important
    features, oversampling, storing/logging results, scoring, and reporting.
The file experiments.py has argparse, initiates one of the Experiment Classes, it has a sklearn pipeline, the 
    classes Reader and Evaluation are called.

"""

# This is how a factory class looks like:

class Shape:
    def draw(self):
        pass

class Circle(Shape):
    def draw(self):
        print("Drawing Circle")

class Rectangle(Shape):
    def draw(self):
        print("Drawing Rectangle")

class ShapeFactory:
    def create_shape(self, shape_type):
        if shape_type == "circle":
            return Circle()
        elif shape_type == "rectangle":
            return Rectangle()
        else:
            raise ValueError("Invalid shape type")

# Usage
factory = ShapeFactory()
circle = factory.create_shape("circle")
rectangle = factory.create_shape("rectangle")

circle.draw()  # Output: Drawing Circle
rectangle.draw()  # Output: Drawing Rectangle

"""
HYPER-PARAMETER TUNING:

There more ways other than using scikit-learn. There are other libraries that perform a fancier hyper-parameter tuning:
- Ray Tune: https://docs.ray.io/en/latest/tune/index.html Works with scikit-learn, XGBoost, transformers
- Optuna: https://optuna.readthedocs.io Works with scikit-learn, XGBoost, but NO transformers
- Hyperopt: https://hyperopt.github.io/hyperopt/ Used to be popular but the other 2 caught up to it and surpassed it

Tutorials of these libraries with Scikit-learn: https://scikit-learn.org:
- https://docs.ray.io/en/latest/tune/examples/tune-sklearn.html#
- https://github.com/optuna/optuna-examples/blob/main/sklearn/sklearn_simple.py
- https://github.com/hyperopt/hyperopt-sklearn

EXPERIMENTAL LOGGING:

When running experiments, you have to deal with an exponential amount of settings and associated results. To help
keep track of what worked under what conditions, there are some tools available. The one recommended is Weights & 
Biases (wandb). To link code from Scikit-learn, PyTorch, HuggingFace libraries, and XGBoost to the interface it needs
a few lines of code.

It provides tooling to store experimental results, models (and their versions)l, hyper-parameter tuning, and 
visualization via Sweeps. 

The quickstart: https://docs.wandb.ai/quickstart

Tutorial to link to Scikit-learn: https://docs.wandb.ai/guides/integrations/scikit
"""