#!/usr/bin/env python
# coding: utf-8

# In[59]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# In[60]:


#Load Test and Training Data
test_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment 2\\cs98x-kannada-mnist\\test.csv')
train_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment 2\\cs98x-kannada-mnist\\training.csv')


# In[61]:


train_data.shape


# In[62]:


#Obseravtion of Training Dataset
train_data.info()


# In[63]:


train_data.isnull().sum().sum()


# In[64]:


train_data.keys()


# In[65]:


train_data.iloc[:, 1]


# In[66]:


#Splitting Target and Predictors
X, Y = train_data.iloc[:, 2:].values, train_data.iloc[:, 1].values


# In[ ]:





# In[67]:


#Distribution of each class
import collections
print(collections.Counter(Y))


# In[ ]:





# In[68]:


#Creating the Training and Test set from data
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42, stratify = Y)


# In[69]:


X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full/255, Y_train_full, test_size = 0.10, random_state = 42, stratify = Y_train_full)


# In[ ]:





# In[70]:


#Feature Scaling
#This is a very important step in machine learning. It helps the algorithm quickly learn a better solution to the problem.
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_valid = scaler.transform(X_valid)
#X_test = scaler.transform(X_test)


# In[71]:


len(X_train_full), len(Y_train_full), len(X_train), len(Y_train), len(X_valid), len(Y_valid)


# In[72]:


print(collections.Counter(Y_train_full))
print(collections.Counter(Y_train))
print(collections.Counter(Y_valid))


# In[73]:


#observation of one image in the mnist data array
some_digit = X_train[1]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap ='binary')
plt.axis('off')
plt.show


# In[74]:


Y_train[1]


# In[75]:


X_train[0].shape


# In[77]:


np.arange(100,400, 100)


# In[79]:


def build_model(n_hidden = 1, n_neurons = 30, learning_rate = 1e-3, optimizer = "sgd", input_shape = X_train[0].shape):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(10, activation = "softmax"))
    #optimizer = keras.optimizers.SGD(lr=learning_rate, nesterov = True)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics = ["accuracy"])
    return model


# In[80]:


keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)


# In[81]:


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
param_distribs = {
    "n_hidden":[0,1,2,3],
    "n_neurons": np.arange(100,400, 100),
    "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05],
    "optimizer": ["sgd", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
}
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True)
rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter = 10, cv = 2)
rnd_search_cv.fit(X_train, Y_train, epochs = 30, validation_data = (X_valid, Y_valid), 
                  callbacks = [early_stopping_cb], verbose=0)


# In[82]:


print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)


# In[ ]:





# In[83]:


kclf = rnd_search_cv.best_estimator_.model


# In[ ]:





# In[ ]:





# In[84]:


kclf.fit(X_train, Y_train, epochs=100, validation_data=(X_valid, Y_valid), verbose=0, callbacks=[early_stopping_cb])


# In[85]:


Y_pred = kclf.predict(X_test)
Y_pred.round(2)


# In[86]:


#print a classification report depicting the precision, recall, and f1-score of t=the different classes and overall model
#from sklearn.metrics import classification_report
#class_rep_rf = classification_report(Y_test, Y_pred)
#print(class_rep_rf)


# In[87]:


# predicting test data results
test_data.keys()


# In[88]:


test_data.iloc[:, 1:]


# In[89]:


#Extracting our features from the test data and predicting their genres
X1 = test_data.iloc[:, 1:].values/255
test_data['label'] = kclf.predict_classes(X1)


# In[90]:


test_data['label'].tail()


# In[91]:


test_data['label'].value_counts()


# In[92]:


#saving predictions as a csv for submission on kaggle
prediction = test_data[['id', 'label']]
prediction.to_csv("KMSAPI21.csv", index=False)
prediction


# In[ ]:





# In[ ]:





# In[ ]:




