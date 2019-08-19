# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:35:36 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\u_datasets\\K_Nearest_Neighbors\\Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x)
x=scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score,KFold
kfold=KFold(n_splits=12,random_state=0)
score=cross_val_score(classifier,x,y,cv=kfold,scoring='accuracy')
score.mean()
print('score:',score.mean())

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

