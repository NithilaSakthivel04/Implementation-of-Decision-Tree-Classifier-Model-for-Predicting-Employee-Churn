# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### Step 1
Prepare your data
Clean and format your data
Split your data into training and testing sets
### Step 2

Define your model
Use a sigmoid function to map inputs to outputs
Initialize weights and bias terms
### Step 3
Define your cost function
Use binary cross-entropy loss function
Penalize the model for incorrect predictions

### Step 4
Define your learning rate
Determines how quickly weights are updated during gradient descent

### Step 5
Train your model
Adjust weights and bias terms using gradient descent
Iterate until convergence or for a fixed number of iterations

### Step 6
Evaluate your model
Test performance on testing data
Use metrics such as accuracy, precision, recall, and F1 score

### Step 7
Tune hyperparameters
Experiment with different learning rates and regularization techniques

### Step 8
Deploy your model
Use trained model to make predictions on new data in a real-world application.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Nithila.S
RegisterNumber: 212224040224
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial data set:

<img width="1024" height="160" alt="Screenshot 2025-10-24 105652" src="https://github.com/user-attachments/assets/03de87b0-1bab-44d9-b692-36315c9171d4" />

### Data info:

<img width="516" height="369" alt="Screenshot 2025-10-24 105701" src="https://github.com/user-attachments/assets/4e6a9882-da62-4720-999a-0d8ceb273040" />

### Optimization of null values:

<img width="286" height="238" alt="image" src="https://github.com/user-attachments/assets/4a14c0cb-9596-48e1-b9c6-0c5c52f2f923" />

### Assignment of x and y values:

<img width="257" height="77" alt="Screenshot 2025-10-24 105726" src="https://github.com/user-attachments/assets/2f2dc140-aceb-42a4-a063-36dcbb823011" />

<img width="1032" height="161" alt="Screenshot 2025-10-24 105734" src="https://github.com/user-attachments/assets/ecd3c478-cd7d-44ad-bc6d-a5d0b4ef4152" />


### Converting string literals to numerical values using label encoder:

<img width="1034" height="173" alt="Screenshot 2025-10-24 105742" src="https://github.com/user-attachments/assets/bf9713b8-7206-44e9-bcd0-f78728146220" />


### Accuracy:

<img width="79" height="30" alt="Screenshot 2025-10-24 105747" src="https://github.com/user-attachments/assets/0bf30bfb-5b5f-4cfd-a328-f2e60395ee46" />

### Prediction:

<img width="1020" height="55" alt="Screenshot 2025-10-24 105753" src="https://github.com/user-attachments/assets/eafd1c62-e9b5-456e-ad21-fbd198ec5baf" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
