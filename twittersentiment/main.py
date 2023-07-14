
#NN dependencies

import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, SimpleRNN, Activation, Dropout, Conv1D
from tensorflow.keras.layers import Embedding, Flatten, LSTM, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

#General dependencies
import pandas as pd
import numpy as np

#Data splitting
from sklearn.model_selection import train_test_split

#For data preprocessing
import spacy
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

import contractions
from bs4 import BeautifulSoup
import re

import tqdm
import unicodedata

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

#Plotting and performance check
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

#Dataset-https://www.kaggle.com/datasets/kazanova/sentiment140
# label is in column 0 with '0'=negative, '4'=positive

data = pd.read_csv(r'C:\Users\arpit\Downloads\training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)

#Shuffle data
shuffled_data = data.sample(frac=1).reset_index(drop=True)

#Truncate the shuffled DataFrame to a given number of rows
num_rows = 50000
truncated_data = shuffled_data[:num_rows]

#Analysing categories
print(truncated_data[3].unique())
print(truncated_data[0].unique())
print(truncated_data[0].value_counts())

#Checking balance of data
sns.countplot(x=0, data=truncated_data)
plt.xlabel("Sentiment")
plt.show()

#NLP tasks
tweets = truncated_data[5]

#label column
y = pd.get_dummies(truncated_data[0]).to_numpy()
print(y)

#Clean data
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_stopwords_and_stemming(text, stem):
    tokens = []
    for token in text.split():
      if token not in stopwords:
        # chops off the ends of words
        if stem:
          tokens.append(stemmer.stem(token))
        else:
          tokens.append(token)
    return " ".join(tokens)

def pre_process_corpus(docs, stem = False, ner=True):
    try:
        # Try loading the 'en_core_web_sm' model
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        # If the model is not found, download it
        print("Downloading 'en_core_web_sm' model...")
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

    norm_docs = []
    # this is to display a progess bar while looping
    for doc in tqdm.tqdm(docs):
        # remove HTML tags
        doc = strip_html_tags(doc)

        # convert tab, new lines to empty spaces
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        # removing URL
        doc = re.sub(r'http\S+', '', doc)

        # lowercasing
        doc = doc.lower()

        # removing accented chars
        doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # expandingshortened words, e.g. don't to do not
        doc = contractions.fix(doc)

        # removing @username
        doc = re.sub('@([A-Za-z0-9_]+)', ' ', doc)

        # Replacing all non alphabets.
        doc = re.sub('[^a-zA-Z]', ' ', doc)

        # Single character removal
        #doc = re.sub(r"\s+[a-zA-Z]\s+", ' ', doc)

        # removing white spaces
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()

        if ner:
          #Applying named entity recognition using spacy
          entities = nlp(doc).ents
          for entity in entities:
            doc = doc.replace(entity.text, entity.label_)

        # removing stop words and apply stemming
        doc = remove_stopwords_and_stemming(doc, stem)

        norm_docs.append(doc)
    return norm_docs

#apply function
cleaned_tweets=pre_process_corpus(tweets)
print(cleaned_tweets)

#Tokenize
# # Print the tokenized tweets
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(cleaned_tweets)

#To sequences
tweet_sequences = tokenizer.texts_to_sequences(cleaned_tweets)

#Some hyperparameters
MAX_LEN=150
EMBED_SIZE=200
batch_size=1000

#Padding

X = pad_sequences(tweet_sequences, maxlen=MAX_LEN, padding="post")

# Split the data into training and test sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

train_sequences.shape

#Accessing glove model
import gensim.downloader as api

# Loading the twitter embeddings model. This model is trained on 2 billion tweets, which contains 27 billion tokens, 1.2 million vocabs.

glove_model = api.load("glove-twitter-200")

# calcultaete number of words
nb_words = len(tokenizer.word_index) + 1
print('All words: ', nb_words)

# obtain the word embedding matrix
embedding_matrix = np.zeros((nb_words, EMBED_SIZE))
for word, i in tokenizer.word_index.items():
    if word in glove_model:
        embedding_matrix[i] = glove_model[word]

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#RNN
def build_model(nb_words, rnn_model="SimpleRNN", embedding_matrix=None):

    model = Sequential()
    # adding an embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(nb_words,
                        EMBED_SIZE,
                        weights=[embedding_matrix],
                        input_length= MAX_LEN,
                        trainable = False))
    else:
        model.add(Embedding(nb_words,
                        EMBED_SIZE,
                        input_length= MAX_LEN,
                        trainable = True))

    # add an RNN layer according to rnn_model
    if rnn_model == "SimpleRNN":
        model.add(SimpleRNN(EMBED_SIZE))
    else:
        model.add(SimpleRNN(EMBED_SIZE))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model


model_rnn = build_model(nb_words, "SimpleRNN", embedding_matrix)
history=model_rnn.fit(train_sequences, train_labels, epochs=20, batch_size=120,
          validation_data=(test_sequences, test_labels), callbacks=EarlyStopping(monitor='val_accuracy', mode='max',patience=3))

# Evaluating the model on the test data
loss, accuracy = model_rnn.evaluate(test_sequences, test_labels)
print(f"Test loss for RNN: {loss}, Test accuracy for RNN: {accuracy}")

#Loss accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Loss function
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



# Sentiment prediction based on user input
def predict_sentiment(model, tokenizer, user_input):
    cleaned_input = pre_process_corpus(user_input)
    input_sequence = tokenizer.texts_to_sequences([cleaned_input])
    input_sequence = pad_sequences(input_sequence, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(input_sequence)
    sentiment_label = np.argmax(prediction)
    if sentiment_label == 0:
        return "Negative sentiment"
    else:
        return "Positive sentiment"

# Prompt user for input and predict sentiment
user_input = input("Enter a sentence: ")
sentiment_prediction = predict_sentiment(model_rnn, tokenizer, user_input)
print(sentiment_prediction)
