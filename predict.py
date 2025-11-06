# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 09:54:53 2025

@author: Sachin Kumar
"""

import joblib
import numpy as np

# Load model
model = joblib.load("expense_model.pkl")

print("---- Monthly Expense Predictor ----")

# User inputs
income = float(input("Enter monthly income: "))
family_size = int(input("Enter family size: "))
spending_score = int(input("Enter spending score (1-10): "))

# Prepare input
X_new = np.array([[income, family_size, spending_score]])

# Predict
predicted_expense = model.predict(X_new)[0]

print(f"Predicted Monthly Expense: ₹{predicted_expense:.2f}")
