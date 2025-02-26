# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1:START

STEP 2:Import the standard Libraries.

STEP 3:Set variables for assigning dataset values.

STEP 4:Import linear regression from sklearn.

STEP 5:Assign the points for representing in the graph.

STEP 6:Predict the regression for marks by using the representation of the graph.

STEP 7:Compare the graphs and hence we obtained the linear regression for the given datas.

STEP 8:END
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHIK KRISHNA TH
RegisterNumber:  212223240067
*/

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SACHIN M
RegisterNumber:  212223040177
*/

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/content/student_scores.csv")

print(df.tail())
print(df.head())
df.info()

x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:,:-1].values   # Scores

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

print("X_Training:", x_train)
print("X_Test:", x_test)
print("Y_Training:", y_train)
print("Y_Test:", y_test)

reg = LinearRegression()
reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

print("Predicted Scores:", Y_pred)
print("Actual Scores:", y_test)

a = Y_pred - y_test
print("Difference (Predicted - Actual):", a)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="red")
plt.title('Training set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
![Screenshot 2025-02-26 091051](https://github.com/user-attachments/assets/fa314eb5-76f3-4196-9dba-98489779ec05)



![Screenshot 2025-02-26 091057](https://github.com/user-attachments/assets/044e31f7-eea7-42d4-8a3b-231ddd8514f0)



![Screenshot 2025-02-26 091104](https://github.com/user-attachments/assets/0c4fe031-2b01-4225-8063-8a98c0e88ac1)



![Screenshot 2025-02-26 091211](https://github.com/user-attachments/assets/5cdcd32c-6969-428a-aaa6-008959024145)



![Screenshot 2025-02-26 091221](https://github.com/user-attachments/assets/732e6eeb-e171-4029-bac7-91fe96e72d26)



![Screenshot 2025-02-26 091229](https://github.com/user-attachments/assets/4994f282-1bad-48a3-b35b-c51367a886c2)



![Screenshot 2025-02-26 091236](https://github.com/user-attachments/assets/ba204f43-b61e-49ce-9c7b-86693a6712a4)



![Screenshot 2025-02-26 091243](https://github.com/user-attachments/assets/79131c8a-7f30-4e5c-8a51-3ea42e5c3928)



![Screenshot 2025-02-26 091252](https://github.com/user-attachments/assets/6cf53c68-e1fe-4df3-aa6d-be80eb6e415a)



![Screenshot 2025-02-26 091259](https://github.com/user-attachments/assets/b4e696f5-8e98-43df-af69-fdce83c643ff)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
