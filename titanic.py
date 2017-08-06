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

"""----------Setear variables para uso en el test--------------"""
passengers_pred=dataset_test["PassengerId"]
y_train = dataset.iloc[:, 1].values


"""---------Dropear columnas que no sirven---------"""
dataset.drop("Cabin",axis=1,inplace=True)
dataset_test.drop("Cabin",axis=1,inplace=True)

dataset.drop(["Survived","PassengerId", "Ticket"],axis=1,inplace=True)
dataset_test.drop(["PassengerId","Ticket"],axis=1,inplace=True)

"""---------FEATURE ENGINEERING----------"""
#Sacar el prefijo a cada nombre
names = dataset['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
names_test= dataset_test['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]

#Crear una nueva columna con los prefijos
dataset["Title"]=names
dataset_test["Title"]=names_test

#Chau nombre
dataset.drop(["Name"],axis=1,inplace=True)
dataset_test.drop(["Name"],axis=1,inplace=True)

#Setear datos faltantes con la mediana
dataset['Fare'] = dataset['Fare'].astype(int)
dataset_test['Fare'].fillna(dataset_test['Fare'].median(), inplace=True)
dataset_test['Fare'] = dataset_test['Fare'].astype(int)
dataset["Age"].fillna(dataset["Age"].median(), inplace=True)
dataset_test["Age"].fillna(dataset_test["Age"].median(), inplace=True)


#Fusiono hermanos y padres y armo un campo que diga si tiene familia o no
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

#Dummies para persona
dataset=pd.get_dummies(dataset ,columns=["Person"], drop_first=True)
dataset_test=pd.get_dummies(dataset_test ,columns=["Person"], drop_first=True)

#Dummies para Pclass
dataset=pd.get_dummies(dataset ,columns=["Pclass"], drop_first=True)
dataset_test=pd.get_dummies(dataset_test ,columns=["Pclass"], drop_first=True)

#Dummies para embarked
dataset=pd.get_dummies(dataset ,columns=["Embarked"], drop_first=True)
dataset_test=pd.get_dummies(dataset_test ,columns=["Embarked"], drop_first=True)

#Dummies para prefijos
train_objs_num = len(dataset)
dataset = pd.concat(objs=[dataset, dataset_test], axis=0)
dataset_preprocessed = pd.get_dummies(dataset, columns=["Title"])
dataset = dataset_preprocessed[:train_objs_num]
dataset_test= dataset_preprocessed[train_objs_num:]

#Seteo si tiene un prefijo raro-de alta sociedad
dataset["Rare"]=dataset["Title_Dona"]+dataset["Title_Lady"]+dataset["Title_the Countess"]+dataset["Title_Capt"]+dataset["Title_Col"]+dataset["Title_Don"]+dataset["Title_Dr"]+dataset["Title_Major"]+dataset["Title_Rev"]+dataset["Title_Sir"]+dataset["Title_Jonkheer"]
dataset_test["Rare"]=dataset_test["Title_Dona"]+dataset_test["Title_Lady"]+dataset_test["Title_the Countess"]+dataset_test["Title_Capt"]+dataset_test["Title_Col"]+dataset_test["Title_Don"]+dataset_test["Title_Dr"]+dataset_test["Title_Major"]+dataset_test["Title_Rev"]+dataset_test["Title_Sir"]+dataset_test["Title_Jonkheer"]

#Chau dummies de prefijo
filter_col = [col for col in list(dataset) if col.startswith('Title_')]
dataset.drop(filter_col, axis=1, inplace=True)
dataset_test.drop(filter_col, axis=1, inplace=True)

"""-----Setear X_train y X_test--------"""
X_train = dataset.iloc[:, :].values
X_test = dataset_test.iloc[:, :].values

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
X_test = sc.transform(X_test)

#RF parece ser el mejor
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

"""-----------------Prediccion final--------------------"""
#Predecir
y_pred_test = classifier.predict(X_test)

"""-----------------Armar csv----------------"""
preds = pd.DataFrame({'PassengerId':passengers_pred, 'Survived':y_pred_test})
preds.to_csv("preds.csv", index=False)
