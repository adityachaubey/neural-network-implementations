# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 07:58:49 2018

@author: adi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 08:55:39 2018

@author: adi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:07:13 2018

@author: adi
"""
def relu(z):
    return z*(z>0)
def der_relu(z):
    return 1*(z>0)
def sigmoid(z):
    w=-z
    t=1/(1+np.exp(w))
    return t

def der_sigmoid(z):
    return (sigmoid(z)*(1-sigmoid(z)))


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

 #importing data

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelX=LabelEncoder()
x[:,1]=labelX.fit_transform(x[:,1])
labelX2=LabelEncoder()
x[:,2]=labelX2.fit_transform(x[:,2])
onehot=OneHotEncoder(categorical_features=[1])
x=onehot.fit_transform(x).toarray()
x=x[:,1:]

#feature scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x=sc_X.fit_transform(x)

# splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# weight initialisation
w1=np.random.uniform(size=(6,11))
b1=np.random.uniform(size=(6,1))
w2=np.random.uniform(size=(1,6))
b2=np.random.uniform(size=(1,1))
alp=0.1
epoch=5000


for i in range(10):
    # forward prop
    
  z2=x_train.dot(w1.T)+b1.T
  a2=relu(z2)
  z3=a2.dot(w2.T)+b2
  a3=sigmoid(z3)
  y_train=y_train.reshape(-1,1)
  
  # backward prop
  delta3=a3-y_train
  delta2=delta3.dot(w2)*der_relu(z2)
  w2=w2-alp*((delta3.T.dot(a2))/8000)
  w1=w1-alp*((delta2.T.dot(x_train))/8000)
  b2=b2-alp*(np.sum(delta3)/8000)
  t=np.sum(delta2,axis=0)
  t=t.reshape(-1,1)
  b1=b1-alp*(t/8000)
  
  j=y_train*np.log(a3)+(1-y_train)*np.log(1-a3)
  j=-(1/8000)*np.sum(j)
  print(j)


m1=x_train.dot(w1.T)+b1.T
am1=relu(m1)
m2=am1.dot(w2.T)+b2
am2=sigmoid(m2)
acc=(8000-np.abs(np.round(am2)-y_train).sum())/8000
w1=w1.reshape(-1,1)
w2=w2.reshape(-1,1)


