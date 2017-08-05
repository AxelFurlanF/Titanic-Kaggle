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

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

#Dummies para Sexo y Embarque
dataset=pd.get_dummies(dataset ,columns=["Sex","Embarked"], drop_first=True)
dataset_test=pd.get_dummies(dataset_test ,columns=["Sex","Embarked"], drop_first=True)
#Columna 10 es Sexo, Columna 11 y 12 es Embarcado
"""
0:PassengerId
1:Survived
2:Pclass
3:Name
4:Sex
5:Age
6:SibSp
7:Parch
8:Ticket
9:Fare
10:Cabin
11:Embark
--------After Dummies------------
0:PassengerId
1:Survived
2-1:Pclass
3-2:Name
4-3:Age
5-4:SibSp
6-5:Parch p-valor: 0.380
7-6:Ticket
8-7:Fare p-valor: 0.383
9-8:Cabin
10-9:Sex_male
11-10:Embarked_Q
12-11:Embarked_S
"""
X_train = dataset.iloc[:, [2,4,5,10,11,12]].values
y_train = dataset.iloc[:, 1].values



"""Arreglar datos"""
"""Edades"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 1:2])
X_train[:, 1:2] = imputer.transform(X_train[:, 1:2])

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

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier()
classifier2.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_train)
y_pred2 = classifier2.predict(X_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
cm2 = confusion_matrix(y_train, y_pred2)


X_test = dataset_test.iloc[:, [1,3,4,9,10,11]].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, 1:2])
X_test[:, 1:2] = imputer.transform(X_test[:, 1:2])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

y_pred_test = classifier2.predict(X_test)
ids = dataset_test.iloc[:, 0]
preds=np.column_stack((ids,y_pred_test))

preds = pd.DataFrame({'PassengerId':ids, 'Survived':y_pred_test})
preds.to_csv("preds.csv", index=False)
