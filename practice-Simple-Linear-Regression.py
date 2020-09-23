#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv(r'E:\data sets\SalData.csv',encoding='latin1')


# In[3]:


data.head()


# In[4]:


data.corr()


# In[15]:


data.info()


# In[16]:


data.describe


# In[6]:


sns.pairplot(data)


# In[18]:


data.isnull().sum()


# In[19]:


data['YearsExperience'].unique()


# In[23]:


data['Salary'].value_counts()


# In[28]:


data.groupby('YearsExperience').Salary.value_counts()


# In[24]:


sns.distplot(data)


# In[29]:


sns.countplot(x='Salary',data=data)


# In[30]:


sns.heatmap(data)


# In[10]:


plt.hist(data['YearsExperience'])


# In[31]:


data.head()


# In[35]:


X=data.iloc[:,:-1]


# In[36]:


X


# In[38]:


y=data.iloc[:,-1]


# In[41]:


print(X.shape)
y.shape


# # Train-Test Split

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# # Model Buildng

# In[48]:


from sklearn.linear_model import LinearRegression


# In[49]:


model=LinearRegression()


# In[52]:


model.fit(X_train,y_train)


# In[53]:


predictions=model.predict(X_test)


# In[54]:


predictions


# In[55]:


# print the coefficients
print(model.intercept_)
print(model.coef_)


# In[65]:


y_test.shape


# In[62]:


from sklearn.metrics import accuracy_score,


# In[67]:


dt = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
dt


# In[68]:


acc = round(model.score(X_train, y_train) * 100, 2)
print (acc)


# In[76]:


# to predict a single value
print(model.predict([[2]]))


# In[78]:


from sklearn import metrics


# In[79]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




