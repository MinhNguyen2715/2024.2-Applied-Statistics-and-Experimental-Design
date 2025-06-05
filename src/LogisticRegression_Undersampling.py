import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold, cross_validate, cross_val_score, train_test_split)
from collections import Counter
from sklearn.metrics import (make_scorer, average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score,roc_curve,auc, accuracy_score,precision_recall_curve)
from sklearn.datasets import make_classification 
from scipy.stats import uniform, loguniform
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns


# import data

data = pd.read_csv('C:/Users/FPT/Downloads/creditcard.csv')

# data.head(5)

# split data
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2, stratify=y, random_state=42)
print(Counter(y_test))

# Apply undersampling
undersampler = RandomUnderSampler(random_state=42,sampling_strategy=0.1)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

print(Counter(y_train_resampled))
print(X_train_resampled.shape)
print(y_train_resampled.shape)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)

# Trainning withe the class weights
model = LogisticRegression(
    class_weight = 'balance',
    max_iter = 1000,
    solver = 'lbfgs'
)

# Hyperparameter tunning
param_grid = {
    'C': [0.001, 0.005, 0.01, 0.02, 0.05],
    'penalty': ['l1', 'l2','elasticnet'],
    'solver': ['liblinear','lbfgs','sag','saga'],
    'class_weight': [None, 'balanced']
}

model = LogisticRegression(max_iter=1000, random_state=42)

scorer = make_scorer(average_precision_score, needs_proba=True)

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train_resampled)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# evaluate on test set 

print("Best params:", grid_search.best_params_)
print("Best AUPRC score:", grid_search.best_score_)


best_model = grid_search.best_estimator_

y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]

y_predic_test = (y_proba_test > 0.5).astype(int)

print("Test AUPRC:", average_precision_score(y_test, y_proba_test))

labels = {1: "Not_Fraud", 0: "Fraud"}

predic_labels = [labels[x] for x in y_predic_test]
true_labels = [labels[x] for x in y_test]

# build DataFrame
df_results = pd.DataFrame(X_test)
df_results["True_label"] = y_test.values
df_results["True_fraud"] = true_labels
df_results["Predic_label"] = y_proba_test
df_results["Predic_fraud"] = predic_labels

print(df_results.head())

# Precision@K function
def precision_at_k(y_true, y_scores, k):
    k = min(k, len(y_scores))
    indices = np.argsort(y_scores)[::-1]
    top_k = y_true[indices[:k]]
    return np.sum(top_k) / k

p_at_100 = precision_at_k(np.array(y_test), y_proba_test, k=100)
print(f"Precision@100: {p_at_100:.4f}")

# Create confusion matrix 

cfmt = confusion_matrix(y_test, y_predic_test)

labels = [labels[i] for i in sorted(labels)]

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cfmt, annot=True, fmt='d', cmap='Blues',xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Fraud")
plt.ylabel("True Fraud")
plt.tight_layout()
plt.show()

# Create summary table
# AUC-ROC curve 
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

#  Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba_test)
pr_auc = average_precision_score(y_test, y_proba_test)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'PR Curve (AUPRC = {pr_auc:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# Confusion matrix components
tn, fp, fn, tp = cfmt.ravel()

# Compute metrics
accuracy  = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print("Precision: ", precision)
print("Recall: ", sensitivity)
print("F1-Score: ", f1)
print("Accuracy: ", accuracy)
print("NPV: ", npv)
print("FPR: ", fpr)
print("FDR: ", fdr)
print("FNR: ", fnr)








