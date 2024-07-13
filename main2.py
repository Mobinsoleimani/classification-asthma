import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
data = pd.read_csv('asthma_disease_data.csv')
# print(data.tail(10))

X = data[['Age','Gender','BMI','Smoking','PetAllergy','FamilyHistoryAsthma','HistoryOfAllergies','ChestTightness']].values

y = data[['Diagnosis']].values

scale = preprocessing.StandardScaler()
normal = scale.fit_transform(X)


Xtrain,Xtest,ytain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)

#classifier algoritm#
dt = DecisionTreeClassifier(criterion= 'entropy',splitter='best',max_depth=4)
dt.fit(Xtrain,ytain)
yhat = dt.predict(Xtest)

print(yhat)
print('_____________________')
print(ytest)
acc = metrics.accuracy_score(ytest, yhat)
print('Accurecy',acc)