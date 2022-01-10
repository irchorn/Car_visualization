#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pylab


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


# In[3]:


#Import dataset
car_file = 'CAR DETAILS FROM CAR DEKHO.csv'
df_used_cars = pd.read_csv(car_file)
display(df_used_cars.head())


# In[4]:


df_used_cars.shape


# In[5]:


df_used_cars.info()


# In[6]:


columns = list(df_used_cars.columns)
print(columns)


# In[7]:


#Checking for missing values
np.sum(df_used_cars.isnull())


# <b>Exploratory Data Analysis(EDA)</b>

# In[8]:


#Categorical columns
categoricals = [i for i in df_used_cars.columns if df_used_cars.dtypes[i] == 'object']
print("Categorical Columns are: ", *categoricals, sep = '\n')


# In[9]:


#Numerical columns
numericals = [i for i in df_used_cars.columns if df_used_cars.dtypes[i] != 'object']
print("Numerical Columns are: ", *numericals, sep = '\n')


# In[10]:


#Identifying unique values
df_used_cars.nunique(axis=0)


# In[13]:


#Distribution of car year by transmission
plt.figure(figsize = (12,10))
sns.displot(data=df_used_cars, x="year", hue="transmission", col="transmission", bins=10)


# Used cars with manual transmission made between 2010 and 2017 had the highest sales.
# For cars with automatic transmission, most sold were used cars made between 2014 and 2017.

# In[12]:


#Relationship between fuel type and car year
sns.relplot(x='year', y='fuel', kind='scatter', data=df_used_cars)


# Cars running on CNG, LPG, and electric cars were mostly made after 2005.

# <b>Analysis on Categorical Attributes</b>

# In[13]:


#Creating a copy of df_used_cars dataframe
copy_df= pd.concat((df_used_cars[categoricals], df_used_cars[numericals]), axis=1)
copy_df.head()


# In[19]:


#Slicing categorical columns of a dataframe
categor_df = copy_df.loc[:, 'name': 'owner']
categor_df.head()


# In[20]:


#Distribution for each categorical feature
fig = plt.figure(1, (14, 8))

for i,categoricals in enumerate(categor_df.drop(['name'], axis=1).columns):
    ax = plt.subplot(2,2,i+1)
    sns.countplot(categor_df[categoricals], order=categor_df[categoricals].value_counts().index)
    ax.set_xlabel(None)
    ax.set_title(f'Distribution of {categoricals}')
    plt.tight_layout()

plt.show()


# In[21]:


#Relationship between categorical variables seller_type and owner
plt.figure(figsize = (10,8))
sns.catplot(x="seller_type", y="owner", data=categor_df)
plt.show()


# Dealer seller_type has all 5 types of owner.

# In[26]:


df_car_name = df_used_cars.sort_values('name' , ascending=False)[:20]


# In[27]:


df_car_name


# In[58]:


plt.figure(figsize=(18,16))
sns.catplot(x="fuel", y="km_driven", kind="box", data=df_used_cars)
plt.xticks(rotation=75)
plt.show()


# In[50]:


#Violin plot to show the relationship of car name to km_driven
plt.figure(figsize=(20,8))
sns.violinplot(data=df_car_name, x='name', y='km_driven', hue='transmission', palette = 'Reds')
plt.xticks(rotation=75)
plt.show()


# The median km_driven for Volkswagen Vento Petrol Highline AT and Volkswagen Vento Petrol Highline are the same, but lower than other cars.

# <b>Analysis on Numerical Attributes</b>

# In[22]:


#Slice numerical columns of a dataframe
numeric_df = copy_df.loc[:, 'year':]
numeric_df.head()


# In[39]:


#Correlation heatmap
sns.heatmap(numeric_df.corr(method='spearman'), annot = True,  cmap = 'coolwarm')
plt.show()


# In[ ]:




