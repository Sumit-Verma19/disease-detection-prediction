import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('diseasedetect/osetoporosis/osteoporosis.csv')
print("\nOverview of Data: \n", df.head())

#Check for null values
print(df.isnull().sum())

#replacing NaN with 'None' in 'Alcohol Consumption', 'Medical Conditions', 'Medications'
df.fillna({'Alcohol Consumption':'None'}, inplace=True)
df.fillna({'Medical Conditions':'None'}, inplace=True)
df.fillna({'Medications':'None'}, inplace=True)

#drop column ID
df = df.drop(['Id'], axis=1)

################################## EDA ##################################

# UNIVARIATE ANALYSIS
fig, ax = plt.subplots(3, 3, figsize=(18, 8))
fig.suptitle("UNIVARIATE SUBPLOT")

# Plot 1: Age Distribution
sns.histplot(x='Age', data=df, color='lightblue', ax=ax[0, 0], bins=20)
ax[0, 0].set_title("Age Distribution")
ax[0, 0].set_xlabel("Age")
ax[0, 0].set_ylabel("Count")

# Plot 2: Gender Distribution
sns.countplot(x='Gender', data=df, color='lightgreen', ax=ax[0, 1])
ax[0, 1].set_title("Gender Distribution")
ax[0, 1].set_xlabel("Gender")
ax[0, 1].set_ylabel("Count")

# Plot 3: Hormonal Changes
sns.countplot(x='Hormonal Changes', data=df, color='cyan', ax=ax[0, 2])
ax[0, 2].set_title("Hormonal Changes Distribution")
ax[0, 2].set_xlabel("Hormonal Changes")
ax[0, 2].set_ylabel("Count")

# Plot 4: Physical Activity
sns.countplot(x='Physical Activity', data=df, color='brown', ax=ax[1, 0])
ax[1, 0].set_title("Physical Activity Distribution")
ax[1, 0].set_xlabel("Physical Activity")
ax[1, 0].set_ylabel("Count")

# Plot 5: Calcium Intake
sns.countplot(x='Calcium Intake', data=df, color='yellow', ax=ax[1, 1])
ax[1, 1].set_title("Calcium Intake Distribution")
ax[1, 1].set_xlabel("Calcium Intake")
ax[1, 1].set_ylabel("Count")

# Plot 6: Medical Conditions
sns.countplot(x='Medical Conditions', data=df, color='magenta', ax=ax[1, 2])
ax[1, 2].set_title("Medical Conditions Distribution")
ax[1, 2].set_xlabel("Medical Conditions")
ax[1, 2].set_ylabel("Count")

# Plot 7: Prior Fractures
sns.countplot(x='Prior Fractures', data=df, color='red', ax=ax[2, 0])
ax[2, 0].set_title("Prior Fractures Distribution")
ax[2, 0].set_xlabel("Prior Fractures")
ax[2, 0].set_ylabel("Count")

# Plot 8: Race/Ethnicity
sns.countplot(x='Race/Ethnicity', data=df, color='lightgrey', ax=ax[2, 1])
ax[2, 1].set_title("Race/Ethnicity Distribution")
ax[2, 1].set_xlabel("Race/Ethnicity")
ax[2, 1].set_ylabel("Count")

# Plot 9: Osteoporosis
sns.countplot(x='Osteoporosis', data=df, color='purple', ax=ax[2, 2])
ax[2, 2].set_title("Osteoporosis Distribution")
ax[2, 2].set_xlabel("Osteoporosis")
ax[2, 2].set_ylabel("Count")

plt.tight_layout()
plt.show()

# BIVARIATE ANALYSIS
fig1, ax1 = plt.subplots(2, 2, figsize=(18, 8))
fig1.suptitle("BIVARIATE SUBPLOT")

# Plot 1: Gender and Osteoporosis
sns.countplot(x='Gender', data=df, hue='Osteoporosis', palette='hls', ax=ax1[0, 0])
ax1[0, 0].set_title('Gender and Osteoporosis')
ax1[0, 0].set_xlabel('Gender')
ax1[0, 0].set_ylabel('Count')

# Plot 2: Age and Osteoporosis
sns.boxplot(x='Osteoporosis', y='Age', data=df, ax=ax1[0, 1])
ax1[0, 1].set_title('Age and Osteoporosis')
ax1[0, 1].set_xlabel('Osteoporosis')
ax1[0, 1].set_ylabel('Age')

# Plot 3: Physical Activity and Osteoporosis
sns.countplot(x='Physical Activity', data=df, hue='Osteoporosis', ax=ax1[1, 0])
ax1[1, 0].set_title('Physical Activity and Osteoporosis')
ax1[1, 0].set_xlabel('Physical Activity')
ax1[1, 0].set_ylabel('Count')

# Plot 4: Prior Fractures and Osteoporosis
sns.countplot(x='Prior Fractures', data=df, hue='Osteoporosis', ax=ax1[1, 1])
ax1[1, 1].set_title('Prior Fractures and Osteoporosis')
ax1[1, 1].set_xlabel('Prior Fractures')
ax1[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()


#############################################FEATURE ENGINEERING AND MODEL BUILDING #############################################

# Label Encoding for categorical variables
categorical_columns = ['Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity', 'Body Weight', 'Calcium Intake', 
                       'Vitamin D Intake', 'Physical Activity', 'Smoking', 'Alcohol Consumption', 'Medical Conditions', 
                       'Medications', 'Prior Fractures']

le = preprocessing.LabelEncoder()
for i in categorical_columns:
    df[i] = le.fit_transform(df[i])

# Splitting Dataset
X = df.drop('Osteoporosis', axis=1)
y = df['Osteoporosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("======================================================")
print("Starting Hyperparameter Tuning for Decision Tree...")

# Decision Tree Hyperparameter Tuning
dtree = DecisionTreeClassifier()
dtree_param_grid = {
    'criterion': ['gini'],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [10],
    'random_state': [42]
}
dtree_grid = RandomizedSearchCV(dtree, dtree_param_grid, cv=5, verbose=0, n_jobs=-1)
dtree_grid.fit(X_train, y_train)

# Best Decision Tree
dtree_best = dtree_grid.best_estimator_
d_pred = dtree_best.predict(X_test)

print("........................................................")
print("Hyperparameter Tuning for Decision Tree Finshed...\n")

print("Best Parameters:", dtree_grid.best_params_)
print("\nDecision Tree Model Performance: \n")
print("........................................................\n")
print(f"Accuracy Score: {accuracy_score(y_test, d_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, d_pred))

print("======================================================")
print("Starting Hyperparameter Tuning for Random Forest...")

# Random Forest Hyperparameter Tuning
rfc = RandomForestClassifier()
rfc_param_grid = {
    'criterion': ['gini'],
    'max_depth': [10, 20],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
    'random_state': [0]
}

rfc_grid = RandomizedSearchCV(rfc, rfc_param_grid, verbose=0, cv=5, n_jobs=-1)
rfc_grid.fit(X_train, y_train)

# Best Random Forest
rfc_best = rfc_grid.best_estimator_
rfc_pred = rfc_best.predict(X_test)

print("........................................................")
print("Hyperparameter Tuning for Random Forest Finshed...\n")

print("Best Parameters:", rfc_grid.best_params_)
print("\nRandom Forest Model Performance: \n")
print("........................................................\n")
print(f"Accuracy Score: {accuracy_score(y_test, rfc_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, rfc_pred))

print("======================================================")
print("Starting Hyperparameter Tuning for Logistic Regression...")

# Logistic Regression Hyperparameter Tuning
logreg = LogisticRegression()
logreg_param_grid = {
    'C': [0.1, 1, 10],  
    'penalty': ['l2'], 
    'solver': ['saga'], 
    'max_iter': [5000],     
}
logreg_grid = RandomizedSearchCV(logreg, logreg_param_grid, cv=5, n_jobs=-1, verbose=0)
logreg_grid.fit(X_train, y_train)

# Best Logistic Regression
logreg_best = logreg_grid.best_estimator_
logreg_pred = logreg_best.predict(X_test)

print("........................................................")
print("Hyperparameter Tuning for Logistic Regression Finished..\n")
print("Best Parameters:", logreg_grid.best_params_)
print("\nLogistic Regression Model Performance: \n")
print("........................................................\n")
print(f"Accuracy Score: {accuracy_score(y_test, logreg_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, logreg_pred))

############################################# PERFORMANCE MEASURES PLOTTING ##############################################
# Performance Measures Subplots
fig4 = plt.figure(figsize=(18, 8))
fig4.suptitle("PERFORMANCE MEASURES SUBPLOTS")

# Confusion Matrix Labels
labels = np.array([['TN', 'FP'], ['FN', 'TP']])

# Decision Tree Confusion Matrix
plt.subplot(3, 2, 1)
cm_dt = confusion_matrix(y_test, d_pred)
sns.heatmap(cm_dt, annot=np.asarray([[f'{val}\n{label}' for val, label in zip(row, labels_row)] for row, 
labels_row in zip(cm_dt, labels)]), fmt='', cmap='Purples', cbar=False)
plt.title('Confusion Matrix for Decision Tree')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['No Osteoporosis', 'Osteoporosis'])
plt.yticks([0.5, 1.5], ['No Osteoporosis', 'Osteoporosis'])

# Random Forest Confusion Matrix
plt.subplot(3, 2, 3)
cm_rf = confusion_matrix(y_test, rfc_pred)
sns.heatmap(cm_rf, annot=np.asarray([[f'{val}\n{label}' for val, label in zip(row, labels_row)] for row, 
labels_row in zip(cm_rf, labels)]), fmt='', cmap='Reds', cbar=False)
plt.title('Confusion Matrix for Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['No Osteoporosis', 'Osteoporosis'])
plt.yticks([0.5, 1.5], ['No Osteoporosis', 'Osteoporosis'])

# Logistic Regression Confusion Matrix
plt.subplot(3, 2, 5)
cm_logreg = confusion_matrix(y_test, logreg_pred)
sns.heatmap(cm_logreg,
            annot=np.asarray([[f'{val}\n{label}' for val, label in zip(row, labels_row)] for row, 
labels_row in zip(cm_rf, labels)]), fmt='', cmap='Reds', cbar=False)
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['No Osteoporosis', 'Osteoporosis'])
plt.yticks([0.5, 1.5], ['No Osteoporosis', 'Osteoporosis'])

# Decision Tree ROC Curve
d_pred_proba = dtree_best.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, d_pred_proba)
roc_auc_dt = roc_auc_score(y_test, d_pred_proba)
plt.subplot(3, 2, 2)
plt.plot(fpr_dt, tpr_dt, color='blue', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.title('ROC Curve for Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

# Random Forest ROC Curve
rfc_pred_proba = rfc_best.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rfc_pred_proba)
roc_auc_rf = roc_auc_score(y_test, rfc_pred_proba)
plt.subplot(3, 2, 4)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

# Logistic Regression ROC Curve
logreg_pred_proba = logreg_best.predict_proba(X_test)[:, 1]
fpr_logreg,tpr_logreg,_ = roc_curve(y_test ,logreg_pred_proba)
roc_auc_logreg = roc_auc_score(y_test ,logreg_pred_proba)
plt.subplot(3 ,2 ,6) 
plt.plot(fpr_logreg ,tpr_logreg ,color='orange' ,lw=2,label=f'Logistic Regression (AUC={roc_auc_logreg:.2f})')
plt.plot([0 ,1] ,[0 ,1] ,color='gray' ,lw=1 ,linestyle='--')
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

############################################ MODEL METRICS ##############################################

# Model Accuracy
fig5 = plt.figure(figsize=(18, 8))
fig5.suptitle("Model Metrics")

# Model Accuracy
plt.subplot(2, 2, 1)
models = ['Logistic Regression', 'Random Forest', 'Decision Tree']
accuracy = [accuracy_score(y_test, logreg_pred), accuracy_score(y_test, rfc_pred), accuracy_score(y_test, d_pred)]
sns.barplot(x=models, y=accuracy, color='Red').set_title('Model Accuracy')

# Mean Absolute Error
plt.subplot(2, 2, 2)
mae = [mean_absolute_error(y_test, logreg_pred), mean_absolute_error(y_test, rfc_pred), mean_absolute_error(y_test, d_pred)]
sns.barplot(x=models, y=mae, color='Blue').set_title('Mean Absolute Error')

# Mean Squared Error
plt.subplot(2, 2, 3)
mse = [mean_squared_error(y_test, logreg_pred), mean_squared_error(y_test, rfc_pred), mean_squared_error(y_test, d_pred)]
sns.barplot(x=models, y=mse, color='Orange').set_title('Mean Squared Error')

# R2 Score
plt.subplot(2, 2, 4)
r2 = [r2_score(y_test, logreg_pred), r2_score(y_test, rfc_pred), r2_score(y_test, d_pred)]
sns.barplot(x=models, y=r2, color='Purple').set_title('R² Score')

plt.tight_layout()
plt.show()
