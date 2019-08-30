# -*- coding: utf-8 -*-

"""
Created on Sat Aug 17 21:59:22 2019

@author: prietojo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl

df = pd.read_csv('datos.csv')

latitud = df[['COORDENADA_1']]
altitud = df[['COORDENADA_2']]

df['FECHA'] = pd.to_datetime(df['FECHA'])

tiempo = df[['FECHA']]

bandera_i = df[['BANDERA_INCENDIO']]

pos = df[df['BANDERA_INCENDIO']>0]

neg = df[df['BANDERA_INCENDIO']<1]


pos_clean = df

pos_clean = pos_clean.drop(columns=['ID_TEMPERATURA','AÑO','CAUSA','ESTADO','MUNICIPIO','ID_INCENDIO','CLAVE','COORDENADA_1','COORDENADA_2','COORDENADAS_REFERENCIA','COORDENADAS','TIPO_VEGETACION','TIPO_INCENDIO','FECHA_INICIO','FECHA_EXT','HECTAREAS','TIPO_IMPACTO','DETECCION','LLEGADA','DURACION','COSTOS','FECHA.1','HORA'])


pos_clean = pos_clean.drop(columns=['FECHA'])




"""

    Scalling part 
    
"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict

import numpy as np

#pos_scaled = preprocessing.scale(pos_clean)

#Cleaning dataframe

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

pos_clean = clean_dataset(pos_clean)

scaler = preprocessing.StandardScaler()


X = pos_clean.drop(columns='BANDERA_INCENDIO')
Y = pos_clean['BANDERA_INCENDIO']

columns =X.columns.tolist()

#X = scaler.fit_transform(X)

#X = pd.DataFrame(X)
#X.set_axis(columns,axis='columns')

# Partiendo los datos en test y train 

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4) 

"""
Resampleando

"""

from sklearn.utils import resample

Xr = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
no_i = Xr[Xr['BANDERA_INCENDIO']<1]
incendio = Xr[Xr['BANDERA_INCENDIO']>0]

incendio_upsampled=resample(incendio,replace=True,n_samples=450,random_state=27)

upsampled = pd.concat([no_i, incendio_upsampled])

X_train = upsampled.drop(columns='BANDERA_INCENDIO')
y_train = upsampled['BANDERA_INCENDIO']

"""
Extreme random forest
"""
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0)
clf.fit(X_train,y_train)
#scores = cross_val_score(clf, X_train, y_train, cv=5)
#scores.mean() 

exrf_pred = clf.predict(X_test)

"""
    Hist Gradient boost
"""
from sklearn.ensemble import GradientBoostingClassifier

clf_hb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
y_pred_hb = clf_hb.predict(X_test)

"""
Random Forest

"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)


"""
                 Clasificador sgdc
"""
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train,y_train)


"""
                # Clasificador bayesiano
"""


from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()

y_pred_g = gnb.fit(X_train, y_train)

y_gnb =gnb.predict_proba(X_test)

y_train_pred_g = gnb.predict(X_test)

y_train_pred = cross_val_predict(sgd_clf,X_test, y_test, cv=5)




# Matriz de confusion 


from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score
from sklearn import metrics


cf_mx = confusion_matrix(y_test,y_train_pred)
print(cf_mx)
pl.matshow(cf_mx)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()


cf_b = confusion_matrix(y_test,y_train_pred_g)
print(cf_b)
pl.matshow(cf_b)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()

cf_rf = confusion_matrix(y_test,rfc_pred)
print(cf_rf)
pl.matshow(cf_rf)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()

cf_erf = confusion_matrix(y_test,exrf_pred)
print(cf_erf)

cf_hb = confusion_matrix(y_test,y_pred_hb)
print(cf_hb)

print("Accuracy:", metrics.accuracy_score(y_test, y_train_pred_g))

print("Accuracy:", metrics.accuracy_score(y_test, y_train_pred))

print("Accuracy:", metrics.accuracy_score(y_test, rfc_pred))

print('Recall', recall_score(y_test, y_train_pred_g, average='macro'))

print('Recall', recall_score(y_test, y_train_pred, average='macro'))

print('Recall', recall_score(y_test, rfc_pred, average='macro'))

print('Recall', recall_score(y_test, exrf_pred, average='macro'))

print('Recall', recall_score(y_test, y_pred_hb, average='macro'))



print('Balanced accuracy', balanced_accuracy_score(y_test,y_train_pred_g))

print('Balanced accuracy', balanced_accuracy_score(y_test,y_train_pred))

print('Balanced accuracy', balanced_accuracy_score(y_test,rfc_pred))


clases = ['No incendio','Incendio']

import pc

a = pc.print_confusion_matrix(cf_rf,clases)

