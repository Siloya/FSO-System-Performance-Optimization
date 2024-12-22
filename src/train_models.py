#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Load training and test data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
X_train, y_train = train_data.drop(columns=["target"]), train_data["target"]
X_test, y_test = test_data.drop(columns=["target"]), test_data["target"]

# Initialize models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, random_state=42)
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Train and evaluate models
models = {"Random Forest": rf, "SVM": svm, "Neural Network": nn}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "Classification Report": classification_report(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }

# Save results
pd.DataFrame(results).to_csv("results/model_performance.csv")
print("Model training and evaluation completed. Results saved.")


