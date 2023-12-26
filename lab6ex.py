###### EXERCISE 1 ###############

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df = pd.read_csv('https://surfdrive.surf.nl/files/index.php/s/'\
                 '9ROTj6HWRAlvngn/download',
                 na_values=['[deleted]', '[removed]'])\
                 .dropna().reset_index(drop=True)

analyzer = SentimentIntensityAnalyzer()

sent1 = analyzer.polarity_scores(df.iloc[16].text)

print("{:-<65} {}".format(df.iloc[16].text, str(sent1)))

sent2 = analyzer.polarity_scores(df.iloc[49].text)

print("{:-<65} {}".format(df.iloc[49].text, str(sent2)))

sent3 = analyzer.polarity_scores("This classifier is so not great.")

print("{:-<65} {}".format("This classifier is so not great.", str(sent3)))

sent4 = analyzer.polarity_scores("This classifier is so great. Not.")

print("{:-<65} {}".format("This classifier is so great. Not.", str(sent4)))

###### EXERCISE 2 ###############
from transformers import pipeline
model = pipeline("text-classification", "cardiffnlp/twitter-roberta-base-sentiment")
score1 = model("a piece of very positive text, yay!")  # LABEL_2 = pos
print(score1)

score2 = model("a piece of very negative text, boo!")  # LABEL_0 = neg
print(score2)

score3 = model("a piece of text")                      # LABEL_1 = neutral
print(score3)

score4 = model("This classifier is so great. Not.")
print(score4)

score5 = model("This classifier is so not great.")
print(score5)

###### EXERCISE 3 ###############

from datasets import load_dataset

dataset = load_dataset("sentiment140")

# print(dataset['train']['sentiment'])
# LABELS ON DATASET: 0 = neg, 4 = pos

# MAPPING REQUIRED FOR MODELS
def map_vader(prediction):
    if prediction['compound'] <= 0:
        return 0
    else:
        return 4

def map_trnsf(prediction):
    if prediction[0]['label'] == 'LABEL_0':
        return 0
    else:  # also for neutral
        return 4

# EVALUATE BOTH MODELS WITH THE DATASET
from sklearn.metrics import accuracy_score

ŷ_V = []
ŷ_T = []
for document in dataset['test']['text']:
    ŷ_V.append(map_vader(analyzer.polarity_scores(document)))
    ŷ_T.append(map_trnsf(model(document)))

print(accuracy_score(dataset['test']['sentiment'], ŷ_V))
print(accuracy_score(dataset['test']['sentiment'], ŷ_T))

# Transformer works a bit better

###### EXERCISE 4 ###############

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(df.iloc[26].text)
# displacy.serve(doc, style="ent")

ents = []
for doc in df.iterrows():
    document = nlp(doc[1].text)
    doc_ent = []
    for ent in document.ents:
        doc_ent.append(ent.text)
    ents.append(doc_ent)

print(ents)

import numpy as np

u = pd.get_dummies(pd.DataFrame(ents), prefix='', prefix_sep='')\
    .groupby(level=0, axis=1).sum()

v = u.T.dot(u)
v.values[(np.r_[:len(v)], ) * 2] = 0

from pyvis.network import Network
import networkx as nx

nt = Network('500px', '500px', notebook=False)
nt.from_nx(nx.from_pandas_adjacency(v))

