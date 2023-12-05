import numpy as np
from collections import Counter


class PPMIVectorizer(object):

    def __init__(self, alpha=0.75, context_window=5):
        self.word_freqs = Counter()
        self.context_freqs = Counter()
        self.co_occ_freqs = Counter()
        self.alpha = alpha
        self.context_window = context_window

    def check_tokens(self, document):
        if not isinstance(document, list):
            return document.split(' ')
        else:
            return document

    def get_counts(self, document):
        for window in zip(*[document[i:] for i in
                            range(self.context_window)]):
            middle = int((len(window) - 1) / 2)
            context = window[:middle] + window[middle + 1:]
            word = window[middle]
            self.word_freqs[word] += 1
            for context_word in context:
                self.context_freqs[context_word] += 1
                self.co_occ_freqs[(word, context_word)] += 1

        return self.co_occ_freqs

    def get_ppmi(self):import numpy as np
from collections import Counter


class PPMIVectorizer(object):

    def __init__(self, alpha=0.75, context_window=5):
        self.word_freqs = Counter()
        self.context_freqs = Counter()
        self.co_occ_freqs = Counter()
        self.alpha = alpha
        self.context_window = context_window

    def check_tokens(self, document):
        if not isinstance(document, list):
            return document.split(' ')
        else:
            return document

    def get_counts(self, document):
        for window in zip(*[document[i:] for i in
                            range(self.context_window)]):
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
        sum_total = sum(self.word_freqs.values()) + sum_context

        for (w, c), wc_freq in self.co_occ_freqs.items():
            P_wc = wc_freq / sum_total
            P_w = self.word_freqs[w] / sum_total
            P_alpha_c = (self.context_freqs[c] ** self.alpha /
                         sum_context ** self.alpha)
            ppmi = max(np.log2(P_wc / (P_w * P_alpha_c)), 0)
            self.co_occ_freqs[(w, c)] = ppmi

        return self.co_occ_freqs

    def set_cols(self):
        self.context_cols = {w: i for i, w in enumerate(
            sorted(self.context_freqs.keys()))}
        self.word_cols = {}
        for w, c in self.co_occ_freqs.keys():
            if not self.word_cols.get(w):
                self.word_cols[w] = []
            self.word_cols[w].append((w, c))

    def get_vec(self, word):
        vec = [0.0] * len(self.context_cols)
        try:
            for (w, c) in self.word_cols[word]:
                vec[self.context_cols[c]] = self.co_occ_freqs[(w, c)]
        except KeyError:
            pass
        return vec

    def fit(self, X):
        for document in X:
            self.get_counts(self.check_tokens(document))
        self.get_ppmi()
        self.set_cols()

    def transform(self, X):
        return [np.sum([self.get_vec(w) for w in
                     self.check_tokens(document)]) for document in X]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        sum_context = sum(self.context_freqs.values())
        sum_total = sum(self.word_freqs.values()) + sum_context

        for (w, c), wc_freq in self.co_occ_freqs.items():
            P_wc = wc_freq / sum_total
            P_w = self.word_freqs[w] / sum_total
            P_alpha_c = (self.context_freqs[c] ** self.alpha /
                         sum_context ** self.alpha)
            ppmi = max(np.log2(P_wc / (P_w * P_alpha_c)), 0)
            self.co_occ_freqs[(w, c)] = ppmi

        return self.co_occ_freqs

    def set_cols(self):
        self.context_cols = {w: i for i, w in enumerate(
            sorted(self.context_freqs.keys()))}
        self.word_cols = {}
        for w, c in self.co_occ_freqs.keys():
            if not self.word_cols.get(w):
                self.word_cols[w] = []
            self.word_cols[w].append((w, c))

    def get_vec(self, word):
        vec = [0.0] * len(self.context_cols)
        try:
            for (w, c) in self.word_cols[word]:
                vec[self.context_cols[c]] = self.co_occ_freqs[(w, c)]
        except KeyError:
            pass
        return vec

    def fit(self, X):
        for document in X:
            self.get_counts(self.check_tokens(document))
        self.get_ppmi()
        self.set_cols()

    def transform(self, X):
        return [np.sum([self.get_vec(w) for w in
                        self.check_tokens(document)]) for document in X]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)