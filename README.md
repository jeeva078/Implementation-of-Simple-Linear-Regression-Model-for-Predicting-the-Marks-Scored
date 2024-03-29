# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: jeevanandam m
RegisterNumber:  212222220017
*/


Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Varadaram SK
RegisterNumber:  202223040232

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120232371/edba4363-a15d-435c-b6b7-1edd1c13d876)
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120232371/d3e7de67-ecf6-453c-94e4-4ccab53b97f1)
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120232371/640dcc14-0663-422a-bc79-d6c3177fff84)
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120232371/67c10836-f397-45c6-a862-ce5db12cf773)
![image](https://github.com/Jayakrishnan22003251/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120232371/2489d851-3eef-4864-9aec-9bd98ab6d59b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
