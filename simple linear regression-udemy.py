#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Importing the dataseet

# In[2]:


df=pd.read_csv(r'E:\Udemy\Machine Learning A-Z (Codes and Datasets)\Part 2 - Regression\Section 4 - Simple Linear Regression\Python\Salary_Data.csv',encoding='latin1')


# In[3]:


df.head()


# In[28]:


X=df.iloc[:,:-1]


# In[29]:


X


# In[30]:


Y=df.iloc[:,-1]


# # Splitting Data set into train and test

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[33]:


X_train


# # Training simple linear regression on Training data

# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


sl=LinearRegression()


# In[36]:


sl.fit(X_train,Y_train)


# In[48]:


# print the coefficients
print(sl.intercept_)
print(sl.coef_)


# # Predicting the test result

# In[37]:


predict=sl.predict(X_test)


# In[38]:


predict


# In[41]:


dt = pd.DataFrame({'Actual': Y_test, 'Predicted': predict})
dt


# In[ ]:


from 


# # Visualising the training set result

# In[43]:


plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,sl.predict(X_train),color='blue')
plt.title("Regression implementation")
plt.xlabel("years of experince")
plt.ylabel("salary")


# In[ ]:





# In[47]:


plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,sl.predict(X_train),color='blue')
plt.title("Regression implementation")
plt.xlabel("years of experince")
plt.ylabel("salary")


# In[51]:


# to predict a single value
print(sl.predict([[12]]))


# In[52]:


print(sl.predict([[35]]))


# In[57]:





# In[ ]:




