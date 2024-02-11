#!/usr/bin/env python
# coding: utf-8

# In[40]:


#importing required libraries to import and analyze the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import DistanceMetric
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,recall_score,precision_score


# In[3]:


#importing the required data set
data=pd.read_csv(r"C:\Users\LENOVO\Downloads\apple_quality.csv")


# In[4]:


data1=data.copy()


# In[5]:


#data1.shape


# In[6]:


#info of the data
#data1.info()


# In[7]:


#Finding no. of null values present for each column in the data
data1.isnull().sum()


# In[8]:


#Dropping the null values
data1.drop(4000,axis=0,inplace=True)


# In[9]:


data1["Acidity"]=data1["Acidity"].astype(float)


# In[10]:


#data1.info()


# In[11]:


#data1.describe()


# In[12]:


#data1


# In[13]:


#data1.info()


# In[14]:


#Checking if the dataset is balanced or not
data1["Quality"].value_counts()


# In[15]:


#a=sns.countplot(data=data1,x=data1["Quality"],hue="Quality")
#for i in a.containers:
#    a.bar_label(i)
#plt.show()


# In[ ]:





# In[18]:


#Extracting the feature variables and class variables
fv=data1.iloc[:,1:-1] #feature variable
cv=data1.iloc[:,-1]  #class variable


# In[19]:


#for y in fv.columns:
#    sns.kdeplot(fv[y])
  #  print(y)
   # plt.show()


# In[20]:


#Finding the correlation between all the variable in the dataset
#fv.corr(method="pearson")


# In[21]:


#ax = sns.heatmap(fv.corr(), annot=True)
#ax


# No two feature variables are highly correlated in either way

# In[22]:


#fv.head()


# In[23]:


#cv.head()

cv=cv.map({"good":1,"bad":0})
# In[25]:


#Checking if all the columns follow normal distribution 
#for y in fv.columns:
 #   plt.subplot(111)
   # ss.probplot(fv[y], dist="norm",fit=True , plot=plt)
   # print(y)
    #plt.show()


# In[49]:


#Dividing the dataset to train,test
x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,stratify=cv,random_state=1)


# In[27]:


#x_train.head()


# In[28]:


#Checking if all the columns follow normal distribution 
#for y in x_train.columns:
  #  plt.subplot(111)
   # ss.probplot(x_train[y], dist="norm",fit=True , plot=plt)
   # print(y)
    #plt.show()


# In[41]:


#Pipeline to impute values
num_p=Pipeline([("imputing_n",SimpleImputer()),("scaling",StandardScaler())])


# In[42]:




# In[50]:


ct=ColumnTransformer([("Numerical",num_p,x_train.columns)])


# In[51]:




# In[52]:


final_ppline=Pipeline([("Pre-processing",ct)])


# In[53]:


final_ppline.fit_transform(x_train,y_train)


# In[54]:


gb1=GaussianNB()


# In[55]:


#fitting the model based on training dataset
model=gb1.fit(final_ppline.fit_transform(x_train),y_train)


# In[56]:


#Predicting the class variables for x_test
pred_y=model.predict(x_test)


# In[ ]:





# In[51]:
pickle.dump(final_ppline,open(r"C:\Users\LENOVO\Downloads\app1.pkl","wb"))

pickle.dump(model,open(r"C:\Users\LENOVO\Downloads\final_model.pkl","wb"))


# In[52]:
import streamlit as s
import sklearn
pre=pickle.load(open(r"C:\Users\LENOVO\Downloads\app1.pkl","rb"))
model1=pickle.load(open(r"C:\Users\LENOVO\Downloads\final_model.pkl","rb"))


s.title("Find the quality of the apple")
s.subheader("Please fill the details to check quality")
size=s.number_input("Enter size of apple")
weight=s.number_input("Enter weight")
sweetness=s.number_input("Enter sweetness")
crunchyness=s.number_input("Enter chrunchyness")
juciness=s.number_input("Enter juciness")
ripeness=s.number_input("Enter ripeness")
acidity=s.number_input("Enter acidity")

f_pre=pre.transform(pd.DataFrame([[size,weight,sweetness,crunchyness,juciness,ripeness,acidity]],columns=["Size","Weight","Sweetness","Crunchiness","Juiciness","Ripeness","Acidity"]))
pred=model1.predict(f_pre)

if pred==1:
   x="Good"
else:
   x="Bad"
if s.button("submit"):
   s.write("Apple quality is ", x) 
# In[54]:


#


# In[ ]:




