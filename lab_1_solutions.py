import string

def tokenize(documents):
    tokenized_documents = []
    for document in documents:
        token_list = []
        for token in document.split(' '):
            if token[-1] in string.punctuation:
                token_list += [token[:-1], token[-1]]
            else:
                token_list.append(token)
        tokenized_documents.append(token_list)
    return tokenized_documents


from collections import Counter

def vectorize(documents):
    M = []
    T = sorted(set([token for document in documents for token in document]))
    for document in documents:
        vector = []
        freq_dict = Counter(document)
        for t in T:
            vector.append(freq_dict[t])
        M.append(vector)
    return M, T

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