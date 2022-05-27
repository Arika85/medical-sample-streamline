# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:32:59 2022

@author: Raghu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Import Dataset
Sample = pd.read_excel('C:/Users/Raghu/Desktop/Arika Project_69/medical_sample_streamline_process_360DigiTMG Project/sampla_data_08_05_2022(final).xlsx')

## Data Understanding
Sample.shape        # Returns size (rows, columns) of data
Sample.info()       # To check the schema/data information
Sample.describe()   # Returns statistical information
Sample.columns

# Data Cleansing & Pre-processing
Sample.isnull().sum()         # Null values/Missing values detection
Sample.duplicated().sum()     # Duplicates detection

## Dropping irrelevant Columns/Features/Variables
Sample.drop(["Patient_ID","Test_Booking_Date","Sample_Collection_Date", "Mode_Of_Transport"], axis = 1, inplace = True)

# creating dummies for categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Sample["Patient_Gender"] = le.fit_transform(Sample["Patient_Gender"])
Sample["Test_Name"] = le.fit_transform(Sample["Test_Name"])
Sample["Sample"] = le.fit_transform(Sample["Sample"])
Sample["Way_Of_Storage_Of_Sample"] = le.fit_transform(Sample["Way_Of_Storage_Of_Sample"])
Sample["Cut-off Schedule"] = le.fit_transform(Sample["Cut-off Schedule"])
Sample["Traffic_Conditions"] = le.fit_transform(Sample["Traffic_Conditions"])

Sample.dtypes

# EDA
# Measures of Central Tendency / First Moment Business decision
Sample.mean()
Sample.median()

# Measures of Dispersion / Second Moment Business Decision
Sample.var()
Sample.std()

# Skewness / Third Moment Business Decision
Sample.skew()

# Kurtosis / Fourth Moment Business Decision
Sample.kurt()

# Data Visualization
# Univariate Analysis
Sample.hist()   # histogram

# Bivariate and Multivariate Analysis (pair plot, scatter plot)
# Pair plot visualization
a = sns.pairplot(Sample)
a

# Findings outliers using boxplot
sns.boxplot(data = Sample, orient = "h")
sns.boxplot(data = Sample.iloc[:, :7], orient = "h")
sns.boxplot(data = Sample.iloc[:, 7:17], orient = "h")
    
#Input and Output Split
x = Sample.iloc[:,:-1]
y = Sample.iloc[:,-1]

# To check for counts of Target/Label column
Sample["Reached_On_Time"].value_counts()

## plotting of Target Column
plt.hist(Sample.Reached_On_Time)

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# Classification Model - output/target variable - Reached_On_Time_Y
# Model 1
# Creating RandomForestClassifier Model in ensemble

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100,
                               criterion="gini",
                               min_samples_split=5,
                               min_samples_leaf=3,
                               random_state=100)
model.fit(x_train, y_train)

## Evaluation of Test dat (Prediction)
y_pred = model.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)

result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

# Train Data Accuracy
accuracy_score(y_train, model.predict(x_train))


# saving the model
# importing the model
import pickle

pickle.dump(model, open('modelbest.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('modelbest.pkl', 'rb'))