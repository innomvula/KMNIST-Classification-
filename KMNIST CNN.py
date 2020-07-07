#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Group Assignment 2 - CS 985/CS 988
#Group AH - Peter Janik (201979128), Inno Mvula (201973944), Thom Reynard (201977555)
#April 8th 2020

#The approach used to complete the classification of the Kannada-MNIST handwritten digits applied a class of deep learning neural networks called Convolutional Neural Networks (CNN). CNNs are powerful deep learning networks and are the preferred deep learning method for image classification. Exploring historical Kaggle scores for the non-Kannada MNIST dataset provides significant indication that the aforementioned CNN model can be considered most likely to be appropriate for the dataset. The combination of data augmentation with convolutional, pooling and dense layers form the foundation of a strong, efficient and accurate machine learning model for the Kannada-MNIST dataset


# In[1]:


#Loading GPU for faster training time
get_ipython().run_line_magic('tensorflow_version', '2.x')
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# In[2]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from google.colab import files
import io
from keras.preprocessing.image import ImageDataGenerator


from numpy.random import seed
seed(2)
tf.random.set_seed(2)


# In[ ]:


import os
os.environ['PYTHONHASHSEED'] = str(2)


# In[4]:


#Uploading training Data
uploaded = files.upload()


# In[ ]:


#Saving training data for use in analysis
train_data = pd.read_csv(io.BytesIO(uploaded['training.csv']))


# In[6]:


#Uploading Test data
uploaded = files.upload()


# In[ ]:


#Saving test data to test our model and make predictions
test_data = pd.read_csv(io.BytesIO(uploaded['test.csv']))


# In[8]:


#Observation of shape and size of training data set
train_data.shape


# In[9]:


#Obseravtion of Training Dataset
train_data.info()


# In[10]:


train_data.isnull().sum().sum()


# In[11]:


train_data.keys()


# In[12]:


train_data.iloc[:, 1]


# In[ ]:


#Splitting Target and Predictors
X, Y = train_data.iloc[:, 2:].values/255, train_data.iloc[:, 1].values


# In[ ]:


#Reshaping dataset as CNN works best with higher dimensions
x = X.reshape(X.shape[0],28,28,1)


# In[15]:


#Distribution of each class
import collections
print(collections.Counter(Y))


# In[ ]:





# In[ ]:


#Creating the Training and Test set from data
X_train_full, X_test, Y_train_full, Y_test = train_test_split(x, Y, test_size = 0.10, random_state = 42, stratify = Y)


# In[ ]:


#Splitting training data into a train set and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, test_size = 0.10, random_state = 42, stratify = Y_train_full)


# In[ ]:





# In[18]:


#Exploring the number of observations/rows in each Split
len(X_train_full), len(Y_train_full), len(X_train), len(Y_train), len(X_valid), len(Y_valid)


# In[19]:


print(collections.Counter(Y_train_full))
print(collections.Counter(Y_train))
print(collections.Counter(Y_valid))


# In[20]:


#observation of one image in the mnist data array
some_digit = X_train[1]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap ='binary')
plt.axis('off')
plt.show


# In[21]:


Y_train[1]


# In[22]:


X_train[0].shape


# In[ ]:


#Creating/Building the model using the sequential API
model = keras.Sequential()

#We then build the Convolutional layers. Flatten layers role is to covnvert each input into a 1D array
model.add(keras.layers.Conv2D(64, 3, activation = "relu", padding = "same", input_shape = X_train[0].shape))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, 3, activation = "relu", padding = "same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, 3, activation = "relu", padding = "same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256, 3, activation = "relu", padding = "same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256, 3, activation = "relu", padding = "same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())

#we then add a dense hidden layer with 300 neurons, using the relu activation function
model.add(keras.layers.Dropout(rate = 0.5, seed = 2))
model.add(keras.layers.Dense(900, activation = "relu"))

#finally we add a dense output layer with 10 neurons, one for class
model.add(keras.layers.Dropout(rate = 0.5, seed = 2))
model.add(keras.layers.Dense(10, activation = "softmax"))


# In[24]:


#summary() method displays the model's layers, their names, their output shapes and number of parameters
model.summary()


# In[ ]:


#compiling the model
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = keras.optimizers.Adamax(learning_rate = 0.005, beta_1 = 0.9, beta_2 = 0.999),
              metrics=["accuracy"])


# In[ ]:





# In[ ]:


#
train_gen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.01, height_shift_range = 0.04)
batches = train_gen.flow(X_train, Y_train, batch_size = 32)
val_batches = train_gen.flow(X_valid, Y_valid, batch_size = 32)


# In[28]:


#Training and evaluating the model
history = model.fit(batches, steps_per_epoch=X_train.shape[0]//32, epochs = 30, validation_data = val_batches, 
                                                   validation_steps = X_valid.shape[0]//32)


# In[ ]:





# In[29]:


pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[30]:


model.evaluate(X_test, Y_test)


# In[31]:


Y_pred = model.predict_classes(X_test)
Y_pred


# In[32]:


#print a classification report depicting the precision, recall, and f1-score of t=the different classes and overall model
from sklearn.metrics import classification_report
class_rep_rf = classification_report(Y_test, Y_pred)
print(class_rep_rf)


# In[33]:


# predicting test data results
test_data.keys()


# In[34]:


test_data.iloc[:, 1:]


# In[ ]:


#Extracting our features from the test data and predicting their genres
X1 = test_data.iloc[:, 1:].values/255
x1 = X1.reshape(X1.shape[0],28,28,1)
test_data['label'] = model.predict_classes(x1)


# In[36]:


test_data.iloc[:, 1:]


# In[37]:


test_data['label'].tail()


# In[38]:


#Observation of the frequency of each class after prediction
test_data['label'].value_counts()


# In[39]:


#saving predictions as a csv for submission on kaggle
prediction2 = test_data[['id', 'label']]

from google.colab import files
prediction2.to_csv("KMSAPI117.csv", index=False)
files.download("KMSAPI117.csv")

prediction2


# In[ ]:




