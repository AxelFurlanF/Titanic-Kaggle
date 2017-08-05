# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 23:10:43 2017

@author: axelf
"""

# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

"""Dropear columnas que no sirven"""
dataset.drop("Cabin",axis=1,inplace=True)
dataset_test.drop("Cabin",axis=1,inplace=True)

dataset.drop(["PassengerId", "Ticket"],axis=1,inplace=True)
dataset_test.drop(["Ticket"],axis=1,inplace=True)

"""Arreglar datos"""
names = dataset['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
names_test= dataset_test['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
dataset["Title"]=names
dataset.drop(["Name"],axis=1,inplace=True)


dataset['Fare'] = dataset['Fare'].astype(int)
dataset_test['Fare'].fillna(dataset_test['Fare'].median(), inplace=True)
dataset_test['Fare'] = dataset_test['Fare'].astype(int)
dataset["Age"].fillna(dataset["Age"].median(), inplace=True)
dataset_test["Age"].fillna(dataset_test["Age"].median(), inplace=True)


#Fusiono hermanos y padres y armo un campo que diaa si tiene familia o no
dataset['Family'] = dataset["Parch"] + dataset["SibSp"]
dataset['Family'].loc[dataset['Family'] > 0] = 1
dataset['Family'].loc[dataset['Family'] == 0] = 0

dataset_test['Family'] = dataset_test["Parch"] + dataset_test["SibSp"]
dataset_test['Family'].loc[dataset_test['Family'] > 0] = 1
dataset_test['Family'].loc[dataset_test['Family'] == 0] = 0

#Dropeo las columnas que use en Family
dataset.drop(['SibSp','Parch'], axis=1, inplace=True)
dataset_test.drop(['SibSp','Parch'], axis=1, inplace=True)

#Sexo lo cambio por Persona
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
dataset['Person'] = dataset[['Age','Sex']].apply(get_person,axis=1)
dataset_test['Person'] = dataset_test[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
dataset.drop(['Sex'],axis=1,inplace=True)
dataset_test.drop(['Sex'],axis=1,inplace=True)

dataset=pd.get_dummies(dataset ,columns=["Person"], drop_first=True)
dataset_test=pd.get_dummies(dataset_test ,columns=["Person"], drop_first=True)

#Dummies para Pclass
dataset=pd.get_dummies(dataset ,columns=["Pclass"], drop_first=True)
dataset_test=pd.get_dummies(dataset_test ,columns=["Pclass"], drop_first=True)

dataset=pd.get_dummies(dataset ,columns=["Embarked"], drop_first=True)
dataset_test=pd.get_dummies(dataset_test ,columns=["Embarked"], drop_first=True)


X_train = dataset.iloc[:, 1:].values
y_train = dataset.iloc[:, 0].values

X_test = dataset_test.iloc[:, 1:].values

#Resumida del modelo
#importar modelos estadisticos
"""
import statsmodels.formula.api as sm
#appendear X_0
X_train = np.append(arr =np.ones((len(X_train), 1)).astype(int), values = X_train, axis = 1)
regressor_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
regressor_OLS.summary()
#corriendo el summary con OLS, p.valores: sibsp y parch, con .380 y .383 respectivamente.
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

"""-----------------Proceso para test--------------------"""
#Predecir
y_pred_test = classifier.predict(X_test)

"""-----------------Armar csv----------------"""
ids = dataset_test.iloc[:, 0]
preds=np.column_stack((ids,y_pred_test))

preds = pd.DataFrame({'PassengerId':ids, 'Survived':y_pred_test})
preds.to_csv("preds.csv", index=False)
