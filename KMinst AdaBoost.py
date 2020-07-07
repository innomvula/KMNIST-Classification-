#!/usr/bin/env python
# coding: utf-8

# In[142]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[143]:


#Load Test and Training Data
test_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment 2\\cs98x-kannada-mnist\\test.csv')
train_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment 2\\cs98x-kannada-mnist\\training.csv')


# In[144]:


train_data.shape


# In[145]:


#Obseravtion of Training Dataset
train_data.info()


# In[146]:


train_data.isnull().sum().sum()


# In[147]:


train_data.keys()


# In[148]:


train_data.iloc[:, 1]


# In[149]:


#Splitting Target and Predictors
X, Y = train_data.iloc[:, 2:].values, train_data.iloc[:, 1].values


# In[ ]:





# In[150]:


#Distribution of each class
import collections
print(collections.Counter(Y))


# In[ ]:





# In[151]:


#Creating the Training and Test set from data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = Y)


# In[152]:


len(X_train), len(Y_train)


# In[153]:


print(collections.Counter(Y_train))


# In[ ]:





# In[154]:


#Train and fit our model using the recommended values for our parameters
tree_clf = DecisionTreeClassifier(max_depth = 5, min_samples_split = 1000, class_weight = 'balanced',
                                 criterion = 'gini')
adb = AdaBoostClassifier(tree_clf, algorithm = 'SAMME.R', learning_rate = 0.1)
adb.fit(X_train, Y_train)


# In[155]:


#evaluation
Y_pred = adb.predict(X_test)
# Making the Confusion Matrix
print(pd.crosstab(Y_test, Y_pred, rownames=['Actual values'], colnames=['Predicted values']))


# In[156]:


#print a classification report depicting the precision, recall, and f1-score of t=the different classes and overall model
from sklearn.metrics import classification_report
class_rep_rf = classification_report(Y_test, Y_pred)
print(class_rep_rf)


# In[104]:


# predicting test data results
test_data.keys()


# In[105]:


test_data.iloc[:, 1:]


# In[107]:


#Extracting our features from the test data and predicting their genres
X1 = test_data.iloc[:, 1:]
test_data['label'] = adb.predict(X1)


# In[108]:


test_data['label'].head()


# In[109]:


test_data['label'].value_counts()


# In[110]:


#saving predictions as a csv for submission on kaggle
prediction = test_data[['id', 'label']]
#prediction.to_csv("KMRanFor1.csv", index=False)
prediction


# In[ ]:





# In[ ]:





# In[ ]:




