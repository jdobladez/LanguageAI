######### EXERCISE 1 #######################

import pandas as pd
from tqdm import tqdm

df = pd.read_csv("reddit-5000.csv")

df = pd.read_csv("reddit-5000.csv", na_values=['[deleted]', '[removed]']).dropna()

print(df.head(5))

######### EXERCISE 2 #######################

party_count = df.subreddit.value_counts()

print(party_count)

party_count_norm = df.subreddit.value_counts(normalize=True)

print(party_count_norm)

######### EXERCISE 3a #######################

punctuation_matches = df.text.str.findall('[^\w\s]') # matches any character that is not a word character(\w) or a whitespace character(\s).

tokens = df.text.str.findall('\w+|[^\w\s]') # match either a sequence of word characters or any non-word, non-space character.
print(tokens.head(5))

######### EXERCISE 3b #######################

from collections import Counter
import matplotlib.pyplot as plt

c = Counter()
for token in tokens:
    c += Counter(token)

ax = pd.DataFrame(c.most_common(10000),
                  columns=['word', 'freq']).plot(loglog=True)
plt.show()

######### EXERCISE 3 BONUS #######################

def top_words_per_subreddit(labels, token_df):
    counters = {label: Counter() for label in labels}
    for ix, row in tqdm(df.iterrows()): # tqdm is a Python library that provides a fast, extensible progress bar for loops and other iterable processes in Python.
        counters[row.subreddit] += Counter(row.text)

    top_words = {label: set([w for w, i in c.most_common(5000)]) for label, c in counters.items()}
    unique_words = {}
    for i, set_i, in top_words.items():
        master_set = set()
        for j, set_j in top_words.items():
            if i == j:
                continue
            master_set = master_set.union(set_j)
        unique_words[i] = set_i - master_set
    return unique_words

unique_labels = df.subreddit.unique()

df['text'] = tokens

top_labels = top_words_per_subreddit(unique_labels, df)

# TEST FUNCTION

from numpy.testing import assert_equal  # only have to import this once
from lab_2_solutions import top_words_per_subreddit as \
    top_words_per_subreddit_solution

try:  # NOTE: make sure your function is in the same file / notebook
    assert_equal(top_words_per_subreddit(
                    df.subreddit.unique(), df),
                 top_words_per_subreddit_solution(
                    df.subreddit.unique(), df))
    print("Success!")
except AssertionError:
    print("Solution is not identical:")
    print("Your func output:", top_words_per_subreddit(df.subreddit.unique(), df))
    print("Solutions output:",
          top_words_per_subreddit_solution(df.subreddit.unique(), df))

######### EXERCISE 4 #########################

import spacy
nlp = spacy.load('en_core_web_sm')

print([token.text for token in nlp("This is a text with some words.")])

class Preprocessor(object):

    def __init__(self, method='regex'):
        self.nlp = spacy.load('en_core_web_sm')
        if method == 'regex':
            self.proc = self.regex_tokens
        elif method == 'spacy':
            self.proc = self.spacy_tokens
        elif method == 'spacy-lemma':
            self.proc = self.spacy_lemma

    def regex_tokens(self, X):
        return X.str.findall('\w+|[^\w\s]').to_list()

    def spacy_tokens(self, X):
        return [[token.text for token in nlp(text)] for text in X] # uses nlp pipeline; result in a list of lists, where
        # each inner list contains tokenized text derived from each original text in X

    def spacy_lemma(self, X):
        return [[token.lemma_ for token in nlp(text)] for text in X]

    def transform(self, X):
        return self.proc(X)


# mirror: https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download
df = pd.read_csv('https://surfdrive.surf.nl/files/index.php/s/9ROTj6HWRAlvngn/download',
                 na_values=['[deleted]', '[removed]']).dropna()
proc = Preprocessor(method='regex')
proc.transform(df.text)

# TEST FUNCTION

from numpy.testing import assert_equal  # only have to import this once
from lab_2_solutions import Preprocessor as Preprocessor_solution

method_test = 'regex'  # NOTE: change this!
try:  # NOTE: make sure your class is in the same file / notebook
    assert_equal(Preprocessor(method=method_test).transform(df.text),
                 Preprocessor_solution(method=method_test).transform(df.text))
    print("Success!")
except AssertionError:
    print("Solution is not identical:")
    print("Your func output:",
          Preprocessor(method=method_test).transform(df.text))
    print("Solutions output:",
          Preprocessor_solution(method=method_test).transform(df.text))

