#!/usr/bin/env python
# coding: utf-8

# In[83]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[84]:


#Load Test and Training Data
test_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment 2\\cs98x-kannada-mnist\\test.csv')
train_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment 2\\cs98x-kannada-mnist\\training.csv')


# In[85]:


train_data.shape


# In[86]:


#Obseravtion of Training Dataset
train_data.info()


# In[87]:


train_data.isnull().sum().sum()


# In[88]:


train_data.keys()


# In[89]:


train_data.iloc[:, 1]


# In[90]:


#Splitting Target and Predictors
X, Y = train_data.iloc[:, 2:].values, train_data.iloc[:, 1].values


# In[ ]:





# In[92]:


#Distribution of each class
import collections
print(collections.Counter(Y))


# In[ ]:





# In[93]:


#Creating the Training and Test set from data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = Y)


# In[94]:


len(X_train), len(Y_train)


# In[95]:


print(collections.Counter(Y_train))


# In[ ]:





# In[57]:


#Here we have selected several parameters we would like to tune to get optimal values in each that improve our model
#parameters = {"max_features":, "max_depth":, "min_samples_split":, 'criterion':, 'class_weight':}
#df = DecisionTreeClassifier()
#max_feat = []
#max_dep = []
#mss = []
#crit = ['gini', 'entropy']
#cweight = ['balanced', None]
#for i in range(1, 11, 1):
#    max_feat.append(i)
#max_feat.append(None)
#for l in range(2, 16, 2):
#    max_dep.append(l)
#for m in range(10, 110, 10):
#    mss.append(m)
#max_dep.append(None)
#parameters = {
#    "max_features": max_feat,
#    "max_depth": max_dep,
#    "min_samples_split": mss,
#    'criterion': crit,
#    'class_weight': cweight
#    
#}
#print(parameters)


# In[58]:


#Here we use GridSearch to hypertune our parameters to get optimal values. Gridsearch iterates through all the values
#set for each parameter and returns the best ones
#from sklearn.model_selection import GridSearchCV
#cv = GridSearchCV(df, parameters, cv = 5)
#cv.fit(X_train, Y_train)


# In[59]:


#Function used to print out different parameter combinations and their scores for the base classifier
#def display(results):
#    print(f'Best parameters are: {results.best_params_}')
#    print("\n")
#    mean_score = results.cv_results_['mean_test_score']
#    std_score = results.cv_results_['std_test_score']
#    params = results.cv_results_['params']
#    for mean, std, params in zip(mean_score, std_score, params):
#        if round(mean,3) >= 0.96:
#            print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


# In[60]:


#display(cv)


# In[ ]:





# In[61]:


#Selection of parameters for main classifier in addition to updating the values of the base classifier parameters
#tree_clf = DecisionTreeClassifier(max_depth = 14, max_features = 10, min_samples_split = 10, class_weight = 'balanced',
#                                 criterion = 'entropy')
#ab = AdaBoostClassifier(tree_clf)
#num_est = []
#lrn_rate = []
#for i in range(5, 100, 5):
#    num_est.append(i)
#for l in range(1, 11, 1):
#    lrn_rate.append(l/10)
#parameters2 = {
#    "n_estimators":num_est,
#    "learning_rate": lrn_rate
#    
#}
#print(parameters2)


# In[62]:


#hyperparameter tuning of main classifier
#from sklearn.model_selection import GridSearchCV
#cv2 = GridSearchCV(ab, parameters2, cv = 5)
#cv2.fit(X_train, Y_train)


# In[63]:


#Function used to print out different parameter combinations and their scores for the main classifier
#def display(results):
#    print(f'Best parameters are: {results.best_params_}')
#    print("\n")
#    mean_score = results.cv_results_['mean_test_score']
#    std_score = results.cv_results_['std_test_score']
#    params = results.cv_results_['params']
#    for mean,std,params in zip(mean_score, std_score, params):
#        if round(mean,3) >= 0.983:
#            print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


# In[64]:


#display(cv2)


# In[ ]:





# In[96]:


#Train and fit our model using the recommended values for our parameters
ranfor = RandomForestClassifier()
ranfor.fit(X_train, Y_train)


# In[97]:


#evaluation
Y_pred = ranfor.predict(X_test)
# Making the Confusion Matrix
print(pd.crosstab(Y_test, Y_pred, rownames=['Actual values'], colnames=['Predicted values']))


# In[98]:


#print a classification report depicting the precision, recall, and f1-score of t=the different classes and overall model
from sklearn.metrics import classification_report
class_rep_rf = classification_report(Y_test, Y_pred)
print(class_rep_rf)


# In[99]:


# predicting test data results
test_data.keys()


# In[100]:


test_data.iloc[:, 1:]


# In[101]:


#Extracting our features from the test data and predicting their genres
X1 = test_data.iloc[:, 1:]
test_data['label'] = ranfor.predict(X1)


# In[103]:


test_data['label'].head()


# In[104]:


test_data['label'].value_counts()


# In[105]:


#saving predictions as a csv for submission on kaggle
prediction = test_data[['id', 'label']]
prediction.to_csv("KMRanFor.csv", index=False)
prediction


# In[ ]:





# In[ ]:





# In[ ]:




