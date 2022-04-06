# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:47:35 2021

@author: PARA
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
from sklearn.svm import SVC
from nltk.corpus import stopwords
import pandas as pd
import joblib

groom_df = pd.read_csv("../Data/groom_dataset.csv")

X, y = groom_df['Text'].values, groom_df['Groom'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelSVM = SVC(C=1, kernel='linear', gamma='scale', random_state=0, probability=True)

modelSVM.fit(X_train, y_train)

#cross_val_score(modelSVM, X, y, cv=5, scoring='accuracy').mean()

#print(modelSVM.score(y_vect, y_test))

modelSVM.predict(y_test)

#print(classification_report(y_test, Y_predSVM))

#modelSVM.score(X_test, y_test)

#print(y_scoreSVM)

#msg_prob = modelSVM.predict_proba(TF_IDF.transform(["Go play soccer"]))
#msg_pred = modelSVM.predict(TF_IDF.transform(["Go play soccer"]))
#print(f"El algoritmo SVM clasifica el mensaje como: {msg_pred[0]} con una confianza del {msg_prob[0][0]:.4f}% para NoGroomer y {msg_prob[0][1]:.4f}% para Groomer")

joblib.dump(modelSVM, 'GroomeClassifier.pkl')
#clf = joblib.load('GroomeClassifier.pkl')