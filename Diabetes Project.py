#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


Diabetes_Data = pd.read_csv(r"C:\Users\Baiyekusi Gilbert\Downloads\diabetes_prediction_dataset.csv\diabetes_prediction_dataset.csv")


# In[3]:


Diabetes_Data.head(10)


# In[4]:


Diabetes_Data.dtypes


# In[5]:


# in creating a dummy variable we need to know the unique values in the object column
Diabetes_Data.gender.unique()


# In[6]:


Diabetes_Data.smoking_history.unique()


# In[7]:


Diabetes_Data.describe(include = "all")


# In[8]:


Diabetes_Data.isnull()


# In[9]:


Diabetes_Data.info()


# In[10]:


# Assuming 'gender' is a column in the DataFrame 'diabetes_data'
# Use pd.get_dummies to create dummy variables
gender_dummies = pd.get_dummies(Diabetes_Data.gender)


# In[11]:


gender_dummies


# In[12]:


smoking_history_dummies = pd.get_dummies(Diabetes_Data.smoking_history)
smoking_history_dummies


# In[13]:


Diabetes_Data 


# In[14]:


new_diabetes_data = pd.concat([Diabetes_Data, gender_dummies, smoking_history_dummies], axis=1)
new_diabetes_data    


# In[15]:


new_diabetes_data.columns


# In[16]:


preprocessed_diabetes_data = new_diabetes_data.drop(['gender','Male','smoking_history','current'], axis=1)
preprocessed_diabetes_data


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


# Split the data into features and target variable
X = preprocessed_diabetes_data.drop('diabetes', axis=1)
Y = preprocessed_diabetes_data['diabetes']


# In[19]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# In[24]:


# Predict the target variable for the test set
y_pred = model.predict(X_test_scaled)


# In[25]:


# Calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




