import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import SnowballStemmer
from wordcloud import STOPWORDS, WordCloud

col_names = ['ID','Age','Sex','Diagnosis','Diet' ,'Type','Weight','Height','Renal','DM','Liquid','NGT','LowSalt','SemiSolid','Normal','LowFat','FatFree']
patients = pd.read_csv('PatientsDiets.csv', header=None, names=col_names)
print("Number of rows in data =",patients.shape[0])
print("Number of columns in data =",patients.shape[1])
print("\n")
print("**Sample data:**")
print(patients.head())

categories = list(patients.columns.values)
categories = categories[8:]
print(categories)

data = patients
data = patients.loc[np.random.choice(patients.index, size=2000)]
data.shape

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

data['Diagnosis'] = data['Diagnosis'].str.lower()
data['Diagnosis'] = data['Diagnosis'].apply(cleanHtml)
data['Diagnosis'] = data['Diagnosis'].apply(cleanPunc)
data['Diagnosis'] = data['Diagnosis'].apply(keepAlpha)
data.head()

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data['Diagnosis'] = data['Diagnosis'].apply(stemming)
data.head()


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=42, test_size=0.20, shuffle=True)

print(train.shape)
print(test.shape)


train_text = train['Diagnosis']
test_text = test['Diagnosis']

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['ID','Age','Sex','Type','Weight','Height'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['ID','Age','Sex','Type','Weight','Height'], axis=1)


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier



# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])
for category in categories:
    print('**Processing {} diet...**'.format(category))

    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category].astype(str))

    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category].astype(str), prediction)))
    print("\n")
