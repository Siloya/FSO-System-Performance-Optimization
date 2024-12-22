#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load training data
data = pd.read_csv("data/train.csv")
X = data.drop(columns=["target"])
y = data["target"]

# Feature selection using mutual information
importance = mutual_info_classif(X, y)
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": importance})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# Save feature importance
feature_importance.to_csv("results/feature_importance.csv", index=False)
print("Feature selection completed. Results saved.")

