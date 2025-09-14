# To implement Logistic Regression.  

import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 

data = load_breast_cancer() 
X = pd.DataFrame(data.data, columns=data.feature_names) 
y = pd.Series(data.target) 

print(X.head()) 
print(y.head()) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000) 
model.fit(X_train, y_train) 

y_pred = model.predict(X_test) 

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
print("\nAccuracy Score:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred))