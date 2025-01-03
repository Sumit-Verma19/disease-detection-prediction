import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import seaborn as sns

# Load and prepare data
df = pd.read_csv("diseasedetect/parkinsons/parkdata.csv")

# Separate features and target
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Scale the features
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# Train the model
model = XGBClassifier(random_state=42)
model.fit(x_train, y_train)

# Calculate predictions and probabilities
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)

print("\nModel Performance:")
print("------------------------------------------------------")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_names = df.columns[1:-1]
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Figure with subplots
plt.figure(figsize=(18, 8))

# Feature importance
plt.subplot(2, 2, 1)
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Parkinson's Disease Detection")
plt.gca().invert_yaxis()

# Distribution of cases
plt.subplot(2, 2, 2)
ax = sns.countplot(data=df, x='status', hue='status', palette=['green', 'orange'], legend=False)
plt.xticks([0, 1], ['No Parkinson\'s', 'Parkinson\'s'])
plt.title("Distribution of Parkinson's Disease Cases")
plt.ylabel("Number of Cases")
plt.xlabel("Patients' Status")
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height() + 5),
                ha='center', va='top', fontsize=10, color='black')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.subplot(2, 2, 3)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.subplot(2, 2, 4)
labels = np.array([['TN', 'FP'],
                  ['FN', 'TP']])
sns.heatmap(cm, annot=np.asarray([[f'{val}\n{label}' for val, label in zip(row, label_row)] 
                                  for row, label_row in zip(cm, labels)]),
            fmt='', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['No Parkinson\'s', 'Parkinson\'s'])
plt.yticks([0.5, 1.5], ['No Parkinson\'s', 'Parkinson\'s'])

plt.tight_layout()
plt.show()