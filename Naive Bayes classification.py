#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('C:/Users/annap/anaconda3/libs/Prostate_Cancer.csv')
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 1].values


# In[3]:


dataset.info()


# In[4]:


M = dataset[dataset.diagnosis_result == "M"]


# In[5]:


B = dataset[dataset.diagnosis_result == "B"]


# In[6]:


plt.title("Malignant vs Benign Tumor")
plt.xlabel("radius")
plt.ylabel("perimeter")
plt.scatter(M.radius, M.perimeter, color = "violet", label = "Malignant", alpha = 0.3)
plt.scatter(B.radius, B.perimeter, color = "blue", label = "Benign", alpha = 0.3)
plt.legend()
plt.show()


# In[7]:


dataset.diagnosis_result = [1 if i== "M" else 0 for i in dataset.diagnosis_result]


# In[8]:


x = dataset.drop(["diagnosis_result"], axis = 1)
y = dataset.diagnosis_result.values


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[10]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)


# In[11]:


print("Naive Bayes score: ",nb.score(x_test, y_test)) 

