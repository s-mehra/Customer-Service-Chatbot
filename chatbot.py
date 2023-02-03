#Importing necessary libraries
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


#open json file
with open('intents.json') as file:
    data = json.load(file)



#lists which will containing items from the json file necessary to training the chatbot
training_sentences = []
training_labels = []
labels = []
responses = []

#Updating lists with items in the JSON object
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['response'])

    if intent['tag'] not in labels:
        labels.append[intent['tag']]



num_classes = len(labels)


lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20

oov_token = "<OOV>"

#vectorized the data using tokenization
tokenizer = Tokenizer(num_word=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncate='post', max_len=max_len)



