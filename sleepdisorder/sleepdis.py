import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('diseasedetect/sleepdisorder/Sleep_health_and_lifestyle_dataset.csv')
print("\nOverview of Data: \n", df.head())

############################# DATA CLEANING #############################

#replacing NaN with 'None' in 'Sleep Disorder'
df.fillna({'Sleep Disorder':'None'}, inplace=True)

#replacing normal weight with normal in BMI column
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')

#drop column Person ID
df.drop('Person ID', axis=1, inplace=True)

#splitting BP into Systolic and Diastolic and dropping BP
df['Systolic_BP'] = df['Blood Pressure'].apply(lambda x: x.split('/')[0])
df['Diastolic_BP'] = df['Blood Pressure'].apply(lambda x: x.split('/')[1])
df.drop('Blood Pressure', axis=1, inplace=True)

print("\nUnique Values from Categorical Columns: \n")
print("BMI Category: ",df['BMI Category'].unique())
print("Sleep Disorder: ",df['Sleep Disorder'].unique())

################################## EDA ##################################

# UNIVARIATE

fig, ax = plt.subplots(3, 3, figsize=(18, 8))
fig.suptitle("UNIVARIATE SUBPLOT")

# Plot 1: Gender
sns.countplot(x='Gender', data=df, color='lightblue', ax=ax[0, 0])
ax[0, 0].set_title("Gender Distribution")
ax[0, 0].set_xlabel("Gender")
ax[0, 0].set_ylabel("Count")

# Plot 2: Age
sns.histplot(x='Age', data=df, color='lightgreen', ax=ax[0, 1], bins=10)
ax[0, 1].set_title("Age Distribution")
ax[0, 1].set_xlabel("Age")
ax[0, 1].set_ylabel("Frequency")

# Plot 3: Sleep Duration
sns.histplot(x='Sleep Duration', data=df, color='cyan', ax=ax[0, 2], bins=10)
ax[0, 2].set_title("Sleep Duration Distribution")
ax[0, 2].set_xlabel("Sleep Duration (hours)")
ax[0, 2].set_ylabel("Frequency")

# Plot 4: Quality of Sleep
sns.countplot(x='Quality of Sleep', data=df, color='brown', ax=ax[1, 0])
ax[1, 0].set_title("Quality of Sleep Distribution")
ax[1, 0].set_xlabel("Quality of Sleep (1-10)")
ax[1, 0].set_ylabel("Count")

# Plot 5: Physical Activity Level
sns.histplot(x='Physical Activity Level', data=df, color='Yellow', ax=ax[1, 1], bins=10)
ax[1, 1].set_title("Physical Activity Level Distribution")
ax[1, 1].set_xlabel("Physical Activity Level")
ax[1, 1].set_ylabel("Frequency")

# Plot 6: Stress Level
sns.countplot(x='Stress Level', data=df, color='Magenta', ax=ax[1, 2])
ax[1, 2].set_title("Stress Level Distribution")
ax[1, 2].set_xlabel("Stress Level (1-10)")
ax[1, 2].set_ylabel("Count")

# Plot 7: BMI Category
sns.countplot(x='BMI Category', data=df, color='Red', ax=ax[2, 0])
ax[2, 0].set_title("BMI Category Distribution")
ax[2, 0].set_xlabel("BMI Category")
ax[2, 0].set_ylabel("Count")

# Plot 8: Daily Steps
sns.histplot(x='Daily Steps', data=df, color='lightgrey', ax=ax[2, 1], bins=10)
ax[2, 1].set_title("Daily Steps Distribution")
ax[2, 1].set_xlabel("Daily Steps")
ax[2, 1].set_ylabel("Frequency")

# Plot 9: Sleep Disorder
sns.countplot(x='Sleep Disorder', data=df, color='purple', ax=ax[2, 2])
ax[2, 2].set_title("Sleep Disorder Distribution")
ax[2, 2].set_xlabel("Sleep Disorder Type")
ax[2, 2].set_ylabel("Count")

plt.tight_layout()
plt.show()

# BIVARIATE

# First subplot
fig1, ax1 = plt.subplots(2, 2, figsize=(18,8))
fig1.suptitle("BIVARIATE SUBPLOT #1")

# Plot 1: Gender and Sleep Disorder
sns.countplot(x='Gender', data=df, palette='hls', hue='Sleep Disorder', ax=ax1[0, 0])
ax1[0, 0].set_title('Gender and Sleep Disorder')
ax1[0, 0].set_xlabel('Gender')
ax1[0, 0].set_ylabel('Count')

# Plot 2: Gender and BMI Category
sns.countplot(x='Gender', hue='BMI Category', data=df, palette='Set1', ax=ax1[0, 1])
ax1[0, 1].set_title('Gender and BMI Category')
ax1[0, 1].set_xlabel('Gender')
ax1[0, 1].set_ylabel('BMI Category')

# Plot 3: Occupation and Sleep Disorder
sns.countplot(x='Occupation', data=df, hue='Sleep Disorder', ax=ax1[1, 0])
ax1[1, 0].set_title('Occupation and Sleep Disorder')
ax1[1, 0].set_xlabel('Occupation')
ax1[1, 0].set_ylabel('Count')
ax1[1, 0].tick_params(axis='x', rotation=90)

# Plot 4: Occupation and Stress Level
sns.boxplot(x='Occupation', y='Stress Level', data=df, ax=ax1[1, 1])
ax1[1, 1].set_title('Occupation and Stress Level')
ax1[1, 1].set_xlabel('Occupation')
ax1[1, 1].set_ylabel('Stress Level')
ax1[1, 1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Second subplot
fig2, ax2 = plt.subplots(2, 2, figsize=(18,8))
fig2.suptitle("BIVARIATE SUBPLOT #2")

# Plot 5: Physical Activity Level and Sleep Disorder
sns.countplot(x='Physical Activity Level', hue='Sleep Disorder', data=df, palette='cool', ax=ax2[0, 0])
ax2[0, 0].set_title('Physical Activity Level and Sleep Disorder')
ax2[0, 0].set_xlabel('Physical Activity Level')
ax2[0, 0].set_ylabel('Count')

# Plot 6: BMI Category and Daily Steps
sns.boxplot(x='BMI Category', y='Daily Steps', data=df, ax=ax2[0, 1])
ax2[0, 1].set_title('BMI Category and Daily Steps')
ax2[0, 1].set_xlabel('BMI Category')
ax2[0, 1].set_ylabel('Daily Steps')

# Plot 7: Age and Sleep Duration
sns.lineplot(x='Age', y='Sleep Duration', data=df, ax=ax2[1, 0])
ax2[1, 0].set_title('Age and Sleep Duration')
ax2[1, 0].set_xlabel('Age')
ax2[1, 0].set_ylabel('Sleep Duration')

# Plot 8: Physical Activity Level and Age
sns.scatterplot(x='Physical Activity Level', y='Age', data=df, ax=ax2[1, 1])
ax2[1, 1].set_title('Physical Activity Level and Age')
ax2[1, 1].set_xlabel('Physical Activity Level')
ax2[1, 1].set_ylabel('Age')

plt.tight_layout()
plt.show()

################# FEATURE ENGINEERING AND MODEL BUILDING #################

# Label Encoding for BMI Category and Sleep Disorder
label_encoder = preprocessing.LabelEncoder()
df['BMI Category']= label_encoder.fit_transform(df['BMI Category'])
df['Sleep Disorder']= label_encoder.fit_transform(df['Sleep Disorder'])

# Splitting Dataset
X = df.drop(['Sleep Disorder','Gender','Occupation'], axis=1)
y = df['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementing Decision Tree and its performance
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

d_pred = dtree.predict(X_test)

print("======================================================")
print("\nDecision Tree Model Performance: \n")
print(f"Accuracy Score: {accuracy_score(y_test, d_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, d_pred))

# Implementing RFA and its performance
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print("======================================================")
print("\nRFA Model Performance: \n")
print(f"Accuracy Score: {accuracy_score(y_test, rfc_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, rfc_pred))


###################### PERFORMANCE MEASURES PLOTTING ######################

#multi-class to binary
y_test_binary = (y_test > 0).astype(int)
d_pred_binary = (d_pred > 0).astype(int)
rfc_pred_binary = (rfc_pred > 0).astype(int)

# Performance Measures Subplots
fig4 = plt.figure(figsize=(18, 8))
fig4.suptitle("PERFORMANCE MEASURES SUBPLOTS")

# Decision Tree Confusion Matrix and Actual vs fitted values
plt.subplot(2, 2, 1)
cm_dt = confusion_matrix(y_test_binary, d_pred_binary)
labels = np.array([['TN', 'FP'],
                  ['FN', 'TP']])
sns.heatmap(cm_dt, annot=np.asarray([[f'{val}\n{label}' for val, label in zip(row, label_row)]
                                    for row, label_row in zip(cm_dt, labels)]),
            fmt='', cmap='Purples', cbar=False)
plt.title('Confusion Matrix for Decision Tree')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['No Disorder', 'Sleep Disorder'])
plt.yticks([0.5, 1.5], ['No Disorder', 'Sleep Disorder'])

plt.subplot(2, 2, 2)
ax = sns.kdeplot(y_test, color="r", label="Actual Value")
sns.kdeplot(d_pred, color="b", label="Fitted Values", ax=ax)
plt.title('Actual vs Fitted Values for Sleep Disorder Prediction using Decision Tree')
plt.xlabel('Sleep Disorder')
plt.ylabel('Proportion of People')

# Random Forest Confusion Matrix and Actual vs fitted values
plt.subplot(2, 2, 3)
cm_rf = confusion_matrix(y_test_binary, rfc_pred_binary)
sns.heatmap(cm_rf, annot=np.asarray([[f'{val}\n{label}' for val, label in zip(row, label_row)]
                                    for row, label_row in zip(cm_rf, labels)]),
            fmt='', cmap='Reds', cbar=False)
plt.title('Confusion Matrix for Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0.5, 1.5], ['No Disorder', 'Sleep Disorder'])
plt.yticks([0.5, 1.5], ['No Disorder', 'Sleep Disorder'])

plt.subplot(2, 2, 4)
ax = sns.kdeplot(y_test, color="r", label="Actual Value")
sns.kdeplot(rfc_pred, color="b", label="Fitted Values", ax=ax)
plt.title('Actual vs Fitted Values for Sleep Disorder Prediction using Random Forest')
plt.xlabel('Sleep Disorder')
plt.ylabel('Proportion of People')

plt.tight_layout()
plt.show()


#=====================================================================================================================#