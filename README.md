# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ARAVIND P
RegisterNumber:  212223240054
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
![SVM For Spam Mail Detection](sam.png)
<br>
data
<br>
![Screenshot 2025-05-17 194619](https://github.com/user-attachments/assets/3d93fa53-db9d-4aa5-9dba-0e9ac7464de3)
<br>
data.shape()
<br>
![Screenshot 2025-05-17 194527](https://github.com/user-attachments/assets/f564cdad-d39e-4746-8f96-0a748b7de740)
<br>
x.shape()
<br>
![Screenshot 2025-05-17 194531](https://github.com/user-attachments/assets/078625c0-3e6a-422c-9985-ddc8caccfca0)
<br>
y.shape()
<br>
![Screenshot 2025-05-17 194531](https://github.com/user-attachments/assets/76de7e2f-af98-4222-a8a4-e06436c87a64)
<br>
x_train
<br>
![Screenshot 2025-05-17 194516](https://github.com/user-attachments/assets/6f9cf887-13ab-4091-bb24-18586d7733d2)
<br>
x_train.shape()
<br>
![Screenshot 2025-05-17 194548](https://github.com/user-attachments/assets/a6a24f73-8b80-4a0b-af6e-b05bbd86789e)
<br>
y_pred
<br>
![Screenshot 2025-05-17 194553](https://github.com/user-attachments/assets/443363a6-e5b8-479e-bbdc-5b8e90a0a5c7)
<br>
acc (accuracy)
<br>
![Screenshot 2025-05-17 194557](https://github.com/user-attachments/assets/163d659e-9bf2-405f-8a2a-d417429ee4c4)
<br>
con (confusion matrix)
<br>
![Screenshot 2025-05-17 194602](https://github.com/user-attachments/assets/8f627d2a-dadd-4333-af98-f54231f92eb9)
<br>
cl (classification report)
<br>
![Screenshot 2025-05-17 194606](https://github.com/user-attachments/assets/69b159b7-7bdb-40cd-82ee-f90812be5450)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
