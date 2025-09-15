# To implement Linear Regression. 

%matplotlib inline #for graph open it in google colab

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score 

housing = fetch_california_housing() 
df = pd.DataFrame(housing.data, columns=housing.feature_names) 
df['MedianHouseValue'] = housing.target 

print(df.info())
print("Missing values in dataset:\n", df.isnull().sum())

X = df.drop('MedianHouseValue', axis=1) 
y = df['MedianHouseValue'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

model = LinearRegression() 
model.fit(X_train_scaled, y_train) 

y_pred = model.predict(X_test_scaled) 

print(f"Intercept (a0): {model.intercept_}") 
print(f"First Coefficient (a1): {model.coef_[0]}") 
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}") 
print(f"R² Score: {r2_score(y_test, y_pred):.2f}") 

plt.figure(figsize=(12, 5)) 
plt.subplot(1, 2, 1) 
plt.scatter(y_test, y_pred, alpha=0.5, color='blue') 
plt.xlabel("Actual Median House Value") 
plt.ylabel("Predicted Median House Value") 
plt.title("Actual vs Predicted Values") 
plt.grid(True) 
plt.tight_layout() 
plt.show()
