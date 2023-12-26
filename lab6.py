"""
INFORMATION EXTRACTION & DEEP LEARNING MIXED INTO ONE:

Evaluation is quite important even with pre-trained Named Entity Recognition (NER). Outputs can be quite different
and not always correct. To know how well something will perform, look at:
1. Publication: (if there is one) read associated paper or doc to see what the model was tested on and the error
2. Data: find a labeled dataset (diff from the one it was trained on), apply to the NER, and look at the performance
3. Qualitative: eyeball performance according to things that matter to us; manually feed data to model and look at the
    actual output rather than just metrics.

For Sentimental Analysis (SA), models will be evaluated using data and inspected using qualitative analysis. For NER,
mostly models will be evaluated using qualitative analysis.

Rule-Based Sentiment (Exercise 1):

For the first classifier, a 'sentiment lexicon' will be used. This is an annotated lexicon (vocabulary) with associated
polarity scores (positive, negative, neutral). Popular option is VADER (and the one used in this lab), its documentation
can be found here: https://github.com/cjhutto/vaderSentiment .

Transformer-Based Sentiment (Exercise 2):

HuggingFace has good libraries, one of them is called transformers: https://huggingface.co/docs/transformers/index
It includes many API to train own models, and they are uploaded to ModelHub. Which means we need to find a good
sentiment model and apply it.

ModelHub: https://huggingface.co/models

The model used in Exercise 2 was cardiffnlp/twitter-roberta-base-sentiment and the labels are:
LABEL_0 = negative, LABEL_1 = neutral, and LABEL_2 = positive.

To compare the models, we will use datasets (library from HuggingFace), which also has a Hub:
https://huggingface.co/datasets . We will use the sentiment-140 one in Exercise 3. It is in a dict-type format.

After comparing Rule-Based and Transformer-Based, we know that simple rule-based models are not that bad in
comparison to state-of-the-art NLP systems.

Entities:

To detect mentions of people's names, apply NER. SpaCy provides entity recognition out of the box and visualizes it.

SpaCy NER doc: https://spacy.io/usage/linguistic-features#named-entities

SpaCy Visualizer doc: https://spacy.io/usage/visualizers#ent

Use displacy.serve() to display the entities and their labels in python script, use display.render() in jupyter.

Labels: ORG, GPE, and MONEY. ORG = companies, agencies, institutions. GPE = geopolitical entity.
MONEY = monetary values, including units.

Mapping all the entities into a co-occurrence matrix, helps visualize it better when the text is huge.
"""

