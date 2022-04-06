from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.pipeline import Pipeline
import itertools, spacy, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.svm import SVC
import pandas as pd

def clean_data(msgs):
    docs = list(nlp.pipe(msgs))
    groom_words = []
    for doc in docs:
        groom_words.append([tkn.lemma_.lower() for tkn in doc if tkn and tkn.is_stop == False and len(tkn) > 3])
    words = itertools.chain(*groom_words)
    return list(set(words))

nlp = spacy.load("es_core_news_md")
df = pd.read_csv("dataset_groom.csv")
docs = list(nlp.pipe(df['Text'].values))

stopwords = list(STOP_WORDS)
tfidf = TfidfVectorizer(ngram_range=(2,3), stop_words=stopwords)
clf = SVC(kernel='linear', gamma='auto', probability=True)

X, y = df['Text'].values, df['Groom'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

X_train_vect = tfidf.fit_transform(X_train)
print(f"Datos entrenamiento: {X_train.shape} \t Objetivo entrenamiento: {y_train.shape} \n Datos prueba: {X_test.shape} \t Objetivo prueba: {y_test.shape}")
print(f"Datos entrenamiento vectorizado: {X_train_vect.shape}")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Objetivo prueba: {y_test.shape} \t Objetivo predicción: {y_pred.shape}")

confusion_matrix(y_test, y_pred)
plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.binary)
plt.show()

print(cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean())
print(f"Error absoluto medio [SVM]: {(mean_absolute_error(y_test.astype('int'), y_pred.astype('int'))*100).round(2)}%")
print(f"Puntaje R^2 [SVM]: {(r2_score(y_test.astype('int'), y_pred.astype('int'))*100).round(2)}%")
print("Conjunto de datos:", df.shape, df['Groom'].shape)
print("Conjunto de entrenamiento:", X_train.shape, y_train.shape)
print("Conjunto de prueba:", X_test.shape, y_test.shape)
print("Métricas de evaluación del modelo Suport Vector Machine:\n")
print(f"Exactitud: {(accuracy_score(y_test, y_pred)*100).round(2)}%")
print(f"Precisión: {(precision_score(y_test, y_pred)*100).round(2)}%")
print(f"Recuerdo: {(recall_score(y_test, y_pred)*100).round(2)}%")
print(f"Puntuación F1: {(f1_score(y_test, y_pred)*100).round(2)}%")
print(f"Perdida de clasificación: {(zero_one_loss(y_test, y_pred)*100).round(2)}%")

y_pred_probSVM = clf.predict_proba(X_test)[:,1]

print(f"Puntaje AUC [SVM]: {((roc_auc_score(y_test, y_pred_probSVM))*100).round(2)}%")
plot_roc_curve(clf, X_test, y_test)
plt.title('SVM')
plt.show()

msg_prob = clf.predict_proba(["Es un grandioso lugar y lo recomiendo mucho!"])
msg_pred = clf.predict(["Es un grandioso lugar y lo recomiendo mucho!"])
print(f"El algoritmo SVM clasifica el mensaje como: {msg_pred[0]} con una confianza del {msg_prob[0][1]*100:.2f}% para No Groomer y {msg_prob[0][2]*100:.2f}% para Groomer")

msg_prob = clf.predict_proba(["Me encantaría tener algunas fotos tuyas con la camisa desabotonada."])
msg_pred = clf.predict(["Me encantaría tener algunas fotos tuyas con la camisa desabotonada."])
print(f"El algoritmo SVM clasifica el mensaje como: {msg_pred[0]} con una confianza del {msg_prob[0][1]*100:.2f}% para No Groomer y {msg_prob[0][2]*100:.2f}% para Groomer")

"""def groom_classifier(*msg):
    msg_pred = model.predict_proba(msg)
    if msg_pred[0][2]*100 > 80.0:
        print(f"El mensaje posee un {msg_pred[0][2]*100:.2f}% de posible contenido grooming!")
    else:
        print(f"El mensaje posee un {msg_pred[0][2]*100:.2f}% de posible contenido grooming")

model.predict_proba(['soy demasiado mayor para ti pero no me importaría cogerte.'])[0][2]*100
import joblib
joblib.dump(model, 'GroomerClassifier.pkl')
model = joblib.load('../GroomerClassifier.pkl')
print(model.predict_proba(['Dale está bien, pero luego jugaremos fútbol.'])[0][2]*100)"""