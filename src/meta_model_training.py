#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load base model predictions
rf_probs = pd.read_csv("results/rf_predictions.csv")
svm_probs = pd.read_csv("results/svm_predictions.csv")
nn_probs = pd.read_csv("results/nn_predictions.csv")

# Combine predictions to train meta-model
X_meta = pd.concat([rf_probs, svm_probs, nn_probs], axis=1)
y_meta = pd.read_csv("data/train.csv")["target"]

meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta, y_meta)

# Evaluate meta-model
y_pred_meta = meta_model.predict(X_meta)
y_prob_meta = meta_model.predict_proba(X_meta)[:, 1]

print("Meta-Model AUC:", roc_auc_score(y_meta, y_prob_meta))



