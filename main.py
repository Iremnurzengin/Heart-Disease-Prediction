# İmporting the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Processing
#1 Loading the .cvdv data
heart_data = pd.read_csv('heart.csv')

#print first 5 rows
print (heart_data.head())

#print last 5 rows
print (heart_data.tail())

# number of rows and colomns
print (heart_data.shape)

# getting some info about deta (data9 is float)
print (heart_data.info())

#checking for missing values (there is no missing value in this data set )
print(heart_data.isnull().sum())

#sstatistical measures 
#it gives you count, mean, std ect.
print(heart_data.describe())

#checking the distribution of target 
# 1 means defective heart , 0 means healthy heart 
print(heart_data['target'].value_counts())

#splitting the features and targets 
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)

#splitting training-test datas
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2 )
print(X.shape, X_train.shape, X_test.shape)

#model training 
#logistic regression 
model = LogisticRegression()

#training the logR model with traiing data  
model.fit(X_train, Y_train)

#model evaluation 
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', training_data_accuracy)
# accuracy: 0,85

#accuracy on test data 
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('accuracy test data: ', test_data_accuracy)
# accuracy : 0,81

# Building Predictive System - Random 
input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)

#change the input data to numpy array 
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print (prediction)

if (prediction[0]==0):
  print('the person does not have heart disease')
else:
  print('the person has heart disease')
  

