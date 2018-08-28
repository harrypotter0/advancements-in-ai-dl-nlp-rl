import numpy as np
import matplotlib 
import pandas as pd
import random
import seaborn as sns

import keras
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn import metrics as mt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from scipy import interp

from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import warnings
warnings.filterwarnings("ignore")

# https://www.kaggle.com/c/twitter-sentiment-analysis2
df = pd.read_csv('dataset/train.csv', encoding="ISO-8859-1")
df = df[:20000]

df.info()

X = np.array(df['SentimentText'])
y = np.array(df['Sentiment'])

n_samples = X.shape[0]
n_classes = len(np.unique(y))

print("n_samples: {}".format(n_samples))
print("n_classes: {}".format(n_classes))

NUM_TOP_WORDS = None
MAX_ART_LEN = 1000 # maximum and minimum number of words

tokenizer = Tokenizer(num_words=NUM_TOP_WORDS)
tokenizer.fit_on_texts(df['SentimentText'])
sequences = tokenizer.texts_to_sequences(df['SentimentText'])

word_index = tokenizer.word_index
NUM_TOP_WORDS = len(word_index) if NUM_TOP_WORDS==None else NUM_TOP_WORDS
top_words = min((len(word_index),NUM_TOP_WORDS))
print('Found %s unique tokens. Distilled to %d top words.' % (len(word_index),top_words))

X = pad_sequences(sequences, maxlen=MAX_ART_LEN)

y_ohe = keras.utils.to_categorical(y)
print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', y_ohe.shape)
print(np.max(X))

