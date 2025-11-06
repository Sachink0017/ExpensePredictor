# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 09:53:28 2025

@author: Sachin Kumar
"""

import numpy as np
import pandas as pd

# For reproducibility
np.random.seed(42)

# Number of samples
n_samples = 100000  

# Features
income = np.random.uniform(5000, 200000, n_samples)       # monthly income
family_size = np.random.randint(1, 9, n_samples)          # family size
spending_score = np.random.randint(1, 11, n_samples)      # spending score (1–10)

# Minimum expense (constant d)
d_true = 2000

# Generate expense using a linear formula + noise
expense = (
    0.25 * income +
    1200 * family_size +
    800 * spending_score +
    d_true +
    np.random.normal(0, 2000, n_samples)   # random noise
)

# Create dataframe
df = pd.DataFrame({
    "income": np.round(income, 2),
    "family_size": family_size,
    "spending_score": spending_score,
    "expense": np.round(expense, 2)
})

# Save to CSV
df.to_csv("monthly_expense_dataset_large.csv", index=False)


print(" Dataset created: monthly_expense_dataset_large.csv with", n_samples, "rows")