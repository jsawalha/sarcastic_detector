import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import cleaning_functions as cf
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import bs4 as BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize, WhitespaceTokenizer, wordpunct_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings('ignore')

#Import training and testing data
train_x = np.load('../Sarcasm_detector/clean_data_for_ML/train_x.npy', allow_pickle=True)
test_x = np.load('../Sarcasm_detector/clean_data_for_ML/test_x.npy', allow_pickle=True)

y_train = np.load('../Sarcasm_detector/clean_data_for_ML/y_train.npy', allow_pickle=True)
y_test = np.load('../Sarcasm_detector/clean_data_for_ML/y_test.npy', allow_pickle=True)

#TDIF vectorization
tf_idf = TfidfVectorizer(ngram_range=(1, 3), max_features=90000, min_df=2)

tv_train_x = tf_idf.fit_transform(train_x)
tv_test_x = tf_idf.transform(test_x)

#Models
# grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

# logit = LogisticRegression(max_iter=10000)

# clf=GridSearchCV(logit,grid,cv=5, verbose=1, n_jobs=-1)



clf = LogisticRegression(max_iter=7000)

clf.fit(tv_train_x, y_train)

y_predict = clf.predict(tv_test_x)

print("Training score is:", clf.score(tv_train_x, y_train))
print("Accuracy score is:", accuracy_score(y_test, y_predict))

#Train model on whole dataset
X = np.load('../Sarcasm_detector/clean_data_for_ML/X_ovr.npy', allow_pickle=True)
y = np.load('../Sarcasm_detector/clean_data_for_ML/y_ovr.npy', allow_pickle=True)

#vectorize the whole dataset
tv_x = tf_idf.transform(X)

clf.fit(tv_x, y)

#pickle dump
pickle.dump(tf_idf, open('tf_idf_vectorizer.pkl', 'wb'))
pickle.dump(clf, open('trained_model.pkl', 'wb'))

# Saving the model and vectorizer content
clf = pickle.load(open('trained_model.pkl','rb'))
tf_idf =pickle.load(open('tf_idf_vectorizer.pkl','rb'))











