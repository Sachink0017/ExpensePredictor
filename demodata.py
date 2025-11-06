# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:55:33 2025

@author: Sachin Kumar
"""

import pandas as pd

# Load the CSV file
file_path = "aptitude_questions_grade3_7.csv"  # replace with your path if different
df = pd.read_csv(file_path)

# Display the first 20 rows
print(df.head(20))

# Optional: Display summary info
print("\nDataset Info:")
print(df.info())

# Optional: Display category distribution
print("\nCategory Counts:")
print(df['category'].value_counts())
