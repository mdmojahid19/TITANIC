#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import csv


# In[2]:


data= pd.read_csv("tested.csv")


# In[3]:


data.head()


# In[4]:


print("total servived: \t\t", data.shape[0])


# In[5]:


print("total servived: \t\t", data.shape[1])


# In[7]:


data.shape


# In[8]:


data.dtypes


# In[10]:


data.describe()


# In[11]:


import seaborn as sns
import plotly.express as px


# In[12]:


sns.countplot(x='Survived',data=data)


# In[14]:


sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')


# In[15]:


sns.countplot(x='Survived',hue='Pclass',data=data,palette='rainbow')


# In[16]:


data['Age'].hist(bins=30,color='lightblue')


# In[17]:


sns.heatmap(data.isnull(),yticklabels=False)


# In[18]:


px.box(data,x='Pclass',y='Age',color='Pclass')


# In[19]:


data.loc[(data['Pclass'] == 1) & (data['Age'].isnull()), 'Age'] = 42
data.loc[(data['Pclass'] == 2) & (data['Age'].isnull()), 'Age'] = 26
data.loc[(data['Pclass'] == 3) & (data['Age'].isnull()), 'Age'] = 24


# In[20]:


data=data.drop(columns='Cabin')


# In[21]:


data=data.dropna()


# In[22]:


data.head()


# In[23]:


sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis')


# In[24]:


data['Age'] = data['Age'].astype(int)
data['Fare'] = data['Fare'].astype(int)


# In[25]:


data['Embarked'] = data['Embarked'].map({'Q': 0,'S':1,'C':2}).astype(int)
data['Sex'] = data['Sex'].map( {'female': 1,'male':0}).astype(int)


# In[26]:


datan = data.drop(['PassengerId','Name','Ticket'],axis = 1, inplace= True)


# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[29]:


data.head()


# In[30]:


x= data.drop(['Survived'],axis=1)
y= data['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[31]:


from sklearn.metrics import accuracy_score
#prediction on test data
y_pred = clf.predict(x_test)
#calculation
acc = accuracy_score(y_test,y_pred)
print('Accuracy:', acc)


# In[ ]:




