#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/sample_dataset.csv")

# Data cleaning and normalization
data = data.dropna()  # Remove missing values
normalized_data = (data - data.mean()) / data.std()

# Split data into training, validation, and test sets
train_data, test_data = train_test_split(normalized_data, test_size=0.3, random_state=42)
train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)

print("Data preprocessing completed. Train and test datasets are saved.")

