import nltk 
'''Done -Used For  text preprocessing, tokenization, part-of-speech 
tagging, parsing, sentiment analysis, machine learning for NLP, and more'''

import spacy #open saurce library for advance NLP with simplified interface

import numpy as np # For mathematical operations on multidimensional array
import pandas as pd # For data Analysis and manipulation

from sklearn.model_selection import train_test_split        #ML Library
from sklearn.feature_extraction.text import TfidfVectorizer 
#TF-IDF vectorization technique  assigns a weight to each word baseon TFandIDF.
# fit_transform takes list of text documents as input and returns a  matrix that represents the TF-IDF 
# vectors for each document


from sklearn.svm import SVC 
#It provides a range of options for kernel functions, regularization parameters, 
# and optimization methods, 
from sklearn.metrics import accuracy_score


# Load data and label relevant information
data = pd.read_csv('legal_documents.csv')
# data['parties'] = np.where(
#     data['text'].str.contains('plaintiff|defendant'), 1, 0)
# data['timeline'] = np.where(data['text'].str.contains('on|before|after'), 1, 0)
# data['liabilities'] = np.where(
#     data['text'].str.contains('liable|liability'), 1, 0)
# Preprocess text using nltk and spacy
nltk.download('punkt') # tokenization package
nlp = spacy.load('en_core_web_sm')
data['text'] = data['text'].apply(lambda x: ' '.join(nltk.word_tokenize(x))) #tokenized data
data['text'] = data['text'].apply(
    lambda x: ' '.join([token.lemma_ for token in nlp(x)]))
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data[['parties', 'timeline', 'liabilities']], test_size=0.2, random_state=42)
# Extract features using TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
#takes a list of text documents as input 
# and returns a  matrix that represents the TF-IDF vectors for each document
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
# Train a support vector machine (SVM) model
model = SVC(kernel='linear')
model.fit(X_train, y_train)
# Evaluate model on testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)