#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


get_ipython().run_line_magic('cd', '"E:/PHYTHON/PROJECT 3"')
cf_train=pd.read_csv("counterfeit_train.csv")
cf_test=pd.read_csv("counterfeit_test.csv")


# In[3]:


cf_train.head()


# In[4]:


cf_test.head()


# In[6]:


# lets combine the data for data prep

cf_test['Counterfeit_Sales']=np.nan
cf_train['data']='train'
cf_test['data']='test'
cf_test=cf_test[cf_train.columns]
cf_all=pd.concat([cf_train,cf_test],axis=0)


# In[9]:


cf_all.head()


# In[8]:


cf_all.dtypes


# In[10]:


# Medicine_ID, DistArea_ID : drop
# Counterfeit_Weight : fillna
# Area_Type, Area_City_Type, Area_dist_level, Medicine_Type, SidEffect_Level: dummies


# In[11]:


cf_all.isnull().sum()


# In[12]:


# Notice that only train data is used for imputing missing values in both train and test 

for col in cf_all.columns:
    if (col not in ['Counterfeit_Sales','data'])& (cf_all[col].isnull().sum()>0):
        cf_all.loc[cf_all[col].isnull(),col]=cf_all.loc[cf_all['data']=='train',col].mean()


# In[14]:


cf_all.isnull().sum()


# In[15]:


cf_all.select_dtypes(['object']).columns


# In[16]:


for col in ['SidEffect_Level', 'Area_Type','Medicine_Type','Area_dist_level','Area_City_Type']:
    
    temp=pd.get_dummies(cf_all[col],prefix=col,dtype=float)
    cf_all=pd.concat([temp,cf_all],1)
    cf_all.drop([col],1,inplace=True)


# In[17]:


cf_all.dtypes


# In[19]:


cf_train=cf_all[cf_all['data']=='train']
del cf_train['data']
cf_test=cf_all[cf_all['data']=='test']
cf_test.drop(['Counterfeit_Sales','data'],axis=1,inplace=True)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


cf_train1,cf_train2=train_test_split(cf_train,test_size=0.2,random_state=2)


# In[23]:


x_train1=cf_train1.drop(['Counterfeit_Sales','Medicine_ID','DistArea_ID'],axis=1)
y_train1=cf_train1['Counterfeit_Sales']
x_train2=cf_train2.drop(['Counterfeit_Sales','Medicine_ID','DistArea_ID'],axis=1)
y_train2=cf_train2['Counterfeit_Sales']


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


lm=LinearRegression()


# In[26]:


lm.fit(x_train1,y_train1)


# In[27]:


x_train1.shape


# In[28]:


lm.intercept_


# In[29]:


list(zip(x_train1.columns,lm.coef_))


# In[30]:


predicted_ir=lm.predict(x_train2)


# In[31]:


from sklearn.metrics import mean_absolute_error


# In[33]:


MAE = mean_absolute_error(y_train2,predicted_ir)


# In[34]:


Score = 1-(MAE/1660)


# In[35]:


Score


# We know the tentative performance now, lets build the model on entire training to make prediction on test/production

# In[41]:


x_train=cf_train.drop(['Counterfeit_Sales','Medicine_ID','DistArea_ID'],axis=1)
y_train=cf_train['Counterfeit_Sales']


# In[ ]:





# In[ ]:





# # Ridge  Regression

# In[36]:


from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV


# In[37]:


lambdas=np.linspace(1,100,100)


# In[38]:


params={'alpha':lambdas}


# In[39]:


model=Ridge(fit_intercept=True)


# In[40]:


grid_search=GridSearchCV(model,param_grid=params,cv=10,scoring='neg_mean_absolute_error')


# In[42]:


grid_search.fit(x_train,y_train)


# In[43]:


grid_search.best_estimator_


# In[44]:


grid_search.cv_results_


#  if you want you can now fit a ridge regression model with obtained value of alpha , although there is no need, grid search automatically fits the best estimator on the entire data, you can directly use this to make predictions on test_data. But if you want to look at coefficients , its much more convenient to fit the model with direct function

# Using the report function given below you can see the cv performance of top few models as well, that will the tentative performance

# In[45]:


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[46]:


report(grid_search.cv_results_,100)


# In[47]:


1-(828.228/1660)


# In[ ]:





# In[ ]:





# ## For looking at coefficients

# In[48]:


ridge_model=grid_search.best_estimator_


# In[49]:


ridge_model.fit(x_train,y_train)


# In[50]:


list(zip(x_train1.columns,ridge_model.coef_))


# ## Lasso Regression

# In[59]:


lambdas=np.linspace(1,10,100)
model=Lasso(fit_intercept=True)
params={'alpha':lambdas}


# In[60]:


grid_search=GridSearchCV(model,param_grid=params,cv=10,scoring='neg_mean_absolute_error')


# In[61]:


grid_search.fit(x_train,y_train)


# In[62]:


grid_search.best_estimator_


# you can see that, the best value of alpha comes at the edge of the range that we tried , we should expand the trial range on that side and run this again

# In[63]:


lambdas=np.linspace(8,10,100)
params={'alpha':lambdas}


# In[64]:


grid_search=GridSearchCV(model,param_grid=params,cv=10,scoring='neg_mean_absolute_error')
grid_search.fit(x_train,y_train)


# In[65]:


grid_search.best_estimator_


# In[66]:


report(grid_search.cv_results_,5)


# In[69]:


1-(825.690/1660)


# In[70]:


list(zip(x_train.columns,lasso_model.coef_))


# In[ ]:




