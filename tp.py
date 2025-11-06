# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:42:59 2025

@author: Sachin Kumar
"""
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("monthly_expense_dataset_large.csv")

# Using pandas
#mean_salary_pandas = df["income"].mean()

# Using numpy
mean_salary_numpy = np.mean(df["income"])
mean_family=np.mean(df['family_size'])
mean_expense=np.mean(df['expense'])

print("Mean salary using numpy:", mean_salary_numpy)
print("Mean salary using numpy:", mean_family)
print("Mean salary using numpy:", mean_expense)