
'All dependencies'
# reading from DB
import pymongo


#NN dependencies
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, SimpleRNN, Activation, Dropout, Conv1D
from tensorflow.keras.layers import Embedding, Flatten, GRU
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

from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

import contractions
from bs4 import BeautifulSoup
import re

import tqdm
import unicodedata

#for tokenisation
from tensorflow.keras.preprocessing.text import Tokenizer

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

'Reading data from DB'
'Connection uri'
uri = "mongodb+srv://SubhLodh:subhlodh@cluster0.yc9it6o.mongodb.net/"

# Connecting to the MongoDB Atlas cluster
client = pymongo.MongoClient(uri)

# Access the database and collection
db = client.tweets
collection = db.tweet_data

#creating cursor
cursor = collection.find({}, {'_id': 0})

# Iterating through the data rows
# for document in cursor:
#     print(document)

# Converting the results to a list of dictionaries
data_list = list(cursor)
#df
df = pd.DataFrame(data_list)

# Closing the connection
client.close()

print(df)

# #Analysing categories
sentiments = df ['label'].unique()
sentiment_counts = df ['label'].value_counts()
print(sentiment_counts)

'NLP tasks'
tweets = df ['text']
#labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df ['label'])
y = y_encoded.astype(np.int32)

#Data preprocessing

def strip_html_tags(text):
    if not isinstance(text, float):
        soup = BeautifulSoup(text, "html.parser")
        # Removing 'iframe' and 'script' elements from the soup
        for s in soup(['iframe', 'script']):
            s.extract()
        # Get the stripped text from the soup
        stripped_text = soup.get_text()
        # Replacing multiple newlines with a single newline
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        return stripped_text

    else:
        print("Error: The 'text' does not contain valid HTML content.")


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

def pre_process_corpus(docs, stem=False, ner=True):
    if docs is None:
        return []

    nlp = spacy.load("en_core_web_sm")
    norm_docs = []

    for doc in tqdm.tqdm(docs):
        #print(f"Processing doc: {doc}")
        if doc is not None and isinstance(doc, str):
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

            # removing white spaces
            doc = re.sub(' +', ' ', doc)
            doc = doc.strip()

            # Applying named entity recognition using spacy
            if ner:
                entities = nlp(doc).ents
                for entity in entities:
                    doc = doc.replace(entity.text, entity.label_)

            # removing stop words and apply stemming
            doc = remove_stopwords_and_stemming(doc, stem)

            norm_docs.append(doc)

    return norm_docs


#Applying cleaning function to data
cleaned_tweets=pre_process_corpus(tweets)
#print("Cleaned samplesa: ", cleaned_tweets)

#Tokenize

tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(cleaned_tweets)

#To sequences
tweet_sequences = tokenizer.texts_to_sequences(cleaned_tweets)

#Some hyperparameters
MAX_LEN=15
EMBED_SIZE=200
batch_size=10

#Padding
X = pad_sequences(tweet_sequences, maxlen=MAX_LEN, padding="post")

# Split the data into training and test sets
if len(X) == len(y):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Now, X_train and y_train contain the training samples and labels,
    # while X_test and y_test contain the testing samples and labels.
else:
    print("Error: The number of samples in 'X' and 'y' is inconsistent.")

#Accessing glove model
import gensim.downloader as api

# Load the twitter embeddings model. This model is trained on 2 billion tweets, which contains 27 billion tokens, 1.2 million vocabs.

glove_model = api.load("glove-twitter-200")

# calcultaete number of words
nb_words = len(tokenizer.word_index) + 1
#print('All words: ', nb_words)

# obtain the word embedding matrix
embedding_matrix = np.zeros((nb_words, EMBED_SIZE))
for word, i in tokenizer.word_index.items():
    if word in glove_model:
        embedding_matrix[i] = glove_model[word]

print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


'Defining the model'
def build_model(nb_words, gru_model="GRU", embedding_matrix=None):

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
    if gru_model == "GRU":
        model.add(GRU(EMBED_SIZE))
    else:
        model.add(GRU(EMBED_SIZE))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

'Model training'

model_gru = None
model_gru = build_model(nb_words, "GRU", embedding_matrix)
history=model_gru.fit(x_train, y_train, epochs=20, batch_size=120,
          validation_data=(x_test, y_test), callbacks=EarlyStopping(monitor='val_accuracy', mode='max', patience=3))

# Evaluating the model on the test data
loss, accuracy = model_gru.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

#Loss accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model_accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# #Loss function
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model_loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

#Predictions
def predict_sentiment(model, tokenizer, user_input):
    cleaned_input = pre_process_corpus(user_input)
    input_sequence = tokenizer.texts_to_sequences([cleaned_input])
    input_sequence = pad_sequences(input_sequence, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(input_sequence)
    sentiment_label = np.argmax(prediction)
    if sentiment_label == 1:
        return "Positive sentiment"

    elif sentiment_label == -1:
        return "Negative sentiment"

    else:
        return "Neutral sentiment"

    # User input prompt and sentiment prediction
    user_input = input("Enter a sentence: ")
    sentiment_prediction = predict_sentiment(model_gru, tokenizer, user_input)
    print(sentiment_prediction)

'Saving model'
model_gru.save("model.h5")
