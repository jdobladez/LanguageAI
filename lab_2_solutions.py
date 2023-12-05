from collections import Counter


def top_words_per_subreddit(labels, df):
    counters = {label: Counter() for label in labels}
    for ix, row in df.iterrows():
        counters[row.subreddit] += Counter(row.text)
    # NOTE: gets too messy if we do this in the loop below
    top_words = {label: set([w for w, i in c.most_common(5000)])
                 for label, c in counters.items()}
    unique_words = {}
    for i, set_i in top_words.items():
        master_set = set()
        for j, set_j in top_words.items():
            if i == j:
                continue
            master_set = master_set.union(set_j)
        unique_words[i] = set_i - master_set

    return unique_words


import spacy


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
        return [[token.text for token in nlp(text)] for text in X]

    def spacy_lemma(self, X):
        return [[token.lemma_ for token in nlp(text)] for text in X]

    def transform(self, X):
        return self.proc(X)