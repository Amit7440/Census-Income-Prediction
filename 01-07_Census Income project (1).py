#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
1) Importing all the required libraries
2) Data loading
3) EDA - Data Preprocessing
4) Visualization
5) Label encoding or one hot encoding
6) Model building -> x and y -> train, test
7) Model creation
8) Hyperparameter tuning

'''

The dataset contains information about individuals from various backgrounds and demographic characteristics. Here is a description of the features (columns) in the Census Income data:

1. Age: Age of the individual in years. (Continuous)

2. Workclass: The individual's type of employment. Categories include Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. (Categorical)

3. Education: The highest level of education achieved by the individual. Categories include Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. (Categorical)

4. Education-num: The number of years of education completed. (Continuous)

5. Marital-status: Marital status of the individual. Categories include Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. (Categorical)

6. Occupation: The individual's occupation. Categories include Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. (Categorical)

7. Relationship: Relationship status of the individual. Categories include Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. (Categorical)

8. Race: The individual's race. Categories include White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. (Categorical)

9. Sex: The individual's gender. Categories include Female, Male. (Categorical)

10. Capital-gain: Capital gains for the individual. (Continuous)

11. Capital-loss: Capital losses for the individual. (Continuous)

12. Hours-per-week: The number of hours worked per week. (Continuous)

13. Native-country: The individual's country of origin. Categories include United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. (Categorical)

14. Income: The target variable indicating whether the individual's income exceeds $50,000 annually. Categories include ">50K" and "<=50K". (Categorical)

The goal of using this dataset is typically to build a machine learning model that can predict the income level of individuals based on their demographic and employment-related features.
# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")


# In[22]:


#load the data
df = pd.read_csv("census-income_final.csv", na_values="?" ,skipinitialspace = True)


# In[23]:


df.head()


# In[24]:


df['age']


# In[25]:


df['sex']


# In[26]:


#rename the last column
df = df.rename(columns = {"Unnamed: 14": "annual_income"})


# In[27]:


df.head()


# In[28]:


df.shape


# In[29]:


df.info()


# In[30]:


df.isnull().sum()


# In[17]:


df['workclass'].value_counts()


# In[18]:


df['workclass'].nunique()


# In[19]:


df['workclass'].unique()


# In[20]:


df.loc[df['workclass']=="?"]


# In[21]:


df['occupation'].value_counts()


# In[31]:


df.isnull().sum()


# In[32]:


df.shape


# In[ ]:


#Handle the null values
#1) df.dropna(inplace=True)
#2) df.fillna(0) ->
#    object -> df['column_name'] = df['column_name'].fillna(df['column_name'].mode()[0])
#    numerical -> df['column_name'] = df['column_name'].fillna(df['column_name'].mean())


# In[33]:


df.columns


# In[34]:


for i in df.columns:
    if (df[i].dtype == "object"):
        
        df[i] = df[i].fillna(df[i].mode()[0])
    else:
        
        df[i] = df[i].fillna(df[i].mean())


# In[35]:


df.isnull().sum()


# In[37]:


#duplicate rows
df[df.duplicated()]


# In[38]:


df.duplicated().sum()


# In[39]:


df.shape


# In[40]:


df.drop_duplicates(inplace=True)


# In[41]:


df.shape


# In[42]:


df.info()


# In[43]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() #create an object

for i in df.columns:
    if ((df[i].dtype=="object") & (i != "annual_income")):
        
        df[i] = le.fit_transform(df[i])


# In[44]:


df.head()


# In[45]:


df.info()


# In[48]:


#multicollinearity
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

col_list=[] #numerical in nature
for i in df.columns:
    if((df[i].dtype!='object') & (i!="annual_income")):
        col_list.append(i)
        
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)


# In[49]:


df = df.drop(["native-country"], axis=1)


# In[50]:


#multicollinearity
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

col_list=[] #numerical in nature
for i in df.columns:
    if((df[i].dtype!='object') & (i!="annual_income")):
        col_list.append(i)
        
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)


# In[51]:


df = df.drop(["education-num"],axis=1)


# In[52]:


#multicollinearity
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

col_list=[] #numerical in nature
for i in df.columns:
    if((df[i].dtype!='object') & (i!="annual_income")):
        col_list.append(i)
        
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)


# In[53]:


df = df.drop(["race"],axis=1)


# In[54]:


#multicollinearity
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

col_list=[] #numerical in nature
for i in df.columns:
    if((df[i].dtype!='object') & (i!="annual_income")):
        col_list.append(i)
        
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)


# In[55]:


df  =df.drop(["hours-per-week"],axis=1)


# In[56]:


#multicollinearity
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

col_list=[] #numerical in nature
for i in df.columns:
    if((df[i].dtype!='object') & (i!="annual_income")):
        col_list.append(i)
        
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)


# In[57]:


df  =df.drop(["workclass"],axis=1)


# In[58]:


#multicollinearity
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

col_list=[] #numerical in nature
for i in df.columns:
    if((df[i].dtype!='object') & (i!="annual_income")):
        col_list.append(i)
        
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)


# In[59]:


df  =df.drop(["education"],axis=1)


# In[60]:


#multicollinearity
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

col_list=[] #numerical in nature
for i in df.columns:
    if((df[i].dtype!='object') & (i!="annual_income")):
        col_list.append(i)
        
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['Feature'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)


# In[62]:


x


# In[63]:


y = df['annual_income']


# In[64]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=990)


# In[65]:


##Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)


# In[66]:


test_pred = lr.predict(x_test) #y_test


# In[67]:


test_pred


# In[68]:


y_test


# In[70]:


from sklearn.metrics import *
accuracy_score(y_test, test_pred)


# In[ ]:





# In[71]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
test_pred = dt.predict(x_test)
test_pred


# In[72]:


from sklearn.metrics import *
accuracy_score(y_test, test_pred)


# In[ ]:





# In[73]:


#Hyperparamter tuning using GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[74]:


# Define the hyperparameter grid

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10,15],
    'min_samples_leaf': [1, 2, 4]
}


# In[75]:


dt = DecisionTreeClassifier()


# In[76]:


# Perform grid search with cross-validation
grid_search = GridSearchCV(dt, param_grid, cv=3)


# In[77]:


grid_search.fit(x, y)  # X and   y are your training data and labels


# In[78]:


best_params = grid_search.best_params_
print(best_params)


# In[79]:


best_dt = grid_search.best_estimator_
print(best_dt)


# In[80]:


# Use the best model for predictions or further analysis
predictions = best_dt.predict(x_test) 


# In[81]:


accuracy_score(predictions, y_test)


# In[82]:


confusion_matrix(predictions, y_test)


# In[83]:


print(predictions)


# In[84]:


print(y_test)


# In[ ]:




