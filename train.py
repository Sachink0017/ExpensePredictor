# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 09:47:14 2025

@author: Sachin Kumar
"""


# train.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ----------------- Load Dataset -----------------
data = pd.read_csv("monthly_expense_dataset_large.csv")

# Features and target
X = data[['income', 'family_size', 'spending_score']]
y = data['expense']

# ----------------- Split into Train and Test -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------- Train Linear Regression -----------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------- Save Model -----------------
joblib.dump(model, "expense_model.pkl")
print(" Model trained and saved as expense_model.pkl")
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Intercept (minimum expense d):", model.intercept_)

# ----------------- Predictions -----------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ----------------- Model Evaluation -----------------
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n----- Model Evaluation -----")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# ----------------- Plot: Training Data vs Predicted -----------------


# Sort values by income so the lines don't zig-zag
sorted_idx = np.argsort(X_train['income'])
X_sorted = X_train['income'].iloc[sorted_idx]
y_train_sorted = y_train.iloc[sorted_idx]
y_pred_sorted = y_train_pred[sorted_idx]

plt.figure(figsize=(10,5))

# Actual values line (blue)
plt.plot(X_sorted, y_train_sorted, color='blue', label='Training Actual')

# Predicted values line (red)
plt.plot(X_sorted, y_pred_sorted, color='red', label='Training Predicted')

plt.xlabel("Income")
plt.ylabel("Monthly Expense")
plt.title("Training Data: Actual vs Predicted")
plt.legend()
plt.show()

'''plt.figure(figsize=(10,5))
plt.scatter(X_train['income'], y_train, color='blue', alpha=0.5, label='Training Actual')
plt.scatter(X_train['income'], y_train_pred, color='red', alpha=0.5, label='Training Predicted')
plt.xlabel("Income")
plt.ylabel("Monthly Expense")
plt.title("Training Data: Actual vs Predicted")
plt.legend()
plt.show()'''

# ----------------- Plot: Test Data vs Predicted -----------------
plt.figure(figsize=(10,5))
plt.scatter(X_test['income'], y_test, color='green', alpha=0.5, label='Test Actual')
plt.scatter(X_test['income'], y_test_pred, color='orange', alpha=0.5, label='Test Predicted')
plt.xlabel("Income")
plt.ylabel("Monthly Expense")
plt.title("Test Data: Actual vs Predicted")
plt.legend()
plt.show()