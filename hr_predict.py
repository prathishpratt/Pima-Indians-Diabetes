# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 04:04:36 2020

@author: hp
"""
#HR Attrition prediction
#    
#    This is a classification problem and am planning to solve this problem with the help of logistic reggrestion.
#    
#    Objective
#    To predict if a employee will leave the job or not with the help of the given data and it consits of columns such as
#    satisfaction_level,number_project,promotion_last_5years etc


import pandas as pd                     #data processing
import numpy as ny                      #liner algebra
import matplotlib.pyplot as plt
import seaborn as sns     

hr_main=pd.read_csv(r"C:\Users\hp\Desktop\spyder\HR_comma_sep.csv",na_values=['??','????'])

hr_copy1=hr_main.copy(deep=True)

hr_copy1.head()       #first 5 rows

hr_copy1.size         #size of the df=rows*columns

hr_copy1.info()       #concise summary of the df

ny.unique(hr_copy1["Department"])  #to get the unique dept.

ny.unique(hr_copy1["salary"])      #to get the unique salary ranges



#------- SO PROBABLY THE GIVEN DF IS VERY CLEAN AND ALL SEEMS GOOD --------

#------- NEXT STEP IS DATA VISUALIZATIONS AND FINDING THE CORR. AND TO SEE WHICH VARIABLES TO CONSIDER 
#IN THE BUILING OF A ML MODEL -------

# Performing an analysis of the data in order to get insights that will answering the next
# question: Why the people are lefting the company ?
hr_left=hr_copy1.groupby( 'left' ).mean()

hr_copy1.isnull().sum()     #to check any null values in any columns

pd.crosstab(index=hr_copy1["satisfaction_level"],columns=hr_copy1["left"])

#TO FIND THE CORRELATION BETWEEN COLUMNS
hr_corr=hr_copy1.corr()


# bar plot in seaborn to see the relationship b/w left and salary
sns.countplot(x="salary",data=hr_copy1,hue="left")
#it can be conluded that the less the salary the more likely the person is to leave

#SCSTTER PLOTS B/W LEFT AND OTHER PARAMETERS

plt.scatter(hr_copy1["satisfaction_level"],hr_copy1['left'],c='red')

plt.scatter(hr_copy1["time_spend_company"],hr_copy1['left'],c='red')

plt.scatter(hr_copy1["promotion_last_5years"],hr_copy1['left'],c='red')

plt.scatter(hr_copy1["salary"],hr_copy1['left'],c='red')

sns.countplot(x="Department",data=hr_copy1,hue="left")

# From the data analysis so far we can conclude that we will use following variables as dependant variables in our model
# **Satisfaction Level**
# **Average Monthly Hours**
# **Promotion Last 5 Years**
# **Salary**

parmeters_hr=hr_copy1[[ 'satisfaction_level','average_montly_hours','promotion_last_5years','salary' ]]

# Salary has all text data. It needs to be converted to numbers and we will use dummy 
# variable for that
#Get Dummies to transform Categorical Variables into Boolean

salary_dummies = pd.get_dummies(hr_copy1.salary, prefix="salary")

#now we need to concat the dummy df with the original

hr_concat=pd.concat([hr_copy1,salary_dummies],axis='columns')

#now we need to drop the salary coloumn

hr_concat.drop(['Department'],axis='columns',inplace=True)

#NOW COMES THE MODEL BULIDING PART

# Initialazing the inputs for our model

X = hr_concat

y = hr_copy1.left

#Generation of the test and training data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)

#NOW LOGISTIC REGGRESSION
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

#Now predicting the first 20 values of the test data
predictions = model.predict( X_test.head(20) )

hr_sub_input = X_test.head( 20 )

# mixing results with inputs

columns_new_name = [ "left" ]
hr_results = pd.DataFrame( predictions, columns=columns_new_name )
hr_final = pd.concat([ hr_sub_input.reset_index(drop=True), hr_results  ],axis='columns')

# In the next results you can see a relation very dependent between the people
# with low salarys and low satisfaction are lefting the company !

