import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import time
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from scipy.stats import uniform, loguniform

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


start1 = time.time()
df = pd.read_csv('creditcard.csv')
print(df.head())
end1 = time.time()
print("Read data time: ", round(end1 - start1, 2), "s")

start2 = time.time()
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(Counter(y_test))

undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)


print(Counter(y_train_resampled))
print(X_train_resampled.shape)
print(y_train_resampled.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)

parameters = {
    'n_estimators': range(10, 21),
    'max_depth': [None],
    'min_samples_split': [2, 3]
}


rf = RandomForestClassifier(random_state=42)


random_search = RandomizedSearchCV(rf, parameters,n_iter=5, cv=5, scoring="average_precision")
random_search.fit(X_train_scaled, y_train_resampled)

best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)
print("Best AUPRC score:", random_search.best_score_)

best_model = random_search.best_estimator_

y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]

from sklearn.metrics import average_precision_score
print("Test AUPRC:", average_precision_score(y_test, y_proba_test))
end2 = time.time()
print("Runtime: " , end2 - start2)


from sklearn.metrics import roc_curve, auc, precision_recall_curve

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


# Card Precision@100 (Precision trong top 100 giao dịch có xác suất gian lận cao nhất)
top_k = 100
top_k_indices = np.argsort(y_proba_test)[-top_k:]
top_k_true = y_test.iloc[top_k_indices]

precision_at_k = top_k_true.sum() / top_k
print(f"🎯 Card Precision@100: {precision_at_k:.4f} ({int(top_k_true.sum())} đúng trong top 100)")

# Chuyển xác suất thành nhãn dự đoán với ngưỡng 0.5
y_pred = (y_proba_test >= 0.5).astype(int)


cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()


accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0
precision = tp / (tp + fp) if (tp + fp) else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0
npv = tn / (tn + fn) if (tn + fn) else 0
fpr = fp / (fp + tn) if (fp + tn) else 0
fdr = fp / (tp + fp) if (tp + fp) else 0
fnr = fn / (fn + tp) if (fn + tp) else 0


print(f"\n Accuracy     : {accuracy:.4f}")
print(f" Sensitivity  : {sensitivity:.4f}")
print(f" Specificity  : {specificity:.4f}")
print(f" Precision    : {precision:.4f}")
print(f" F1 Score     : {f1:.4f}")
print(f" NPV          : {npv:.4f}")
print(f" FPR          : {fpr:.4f}")
print(f" FDR          : {fdr:.4f}")
print(f" FNR          : {fnr:.4f}")
