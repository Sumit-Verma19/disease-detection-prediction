import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load and prepare data
data = pd.read_csv('diseasedetect/diabetes/diabetes.csv')

# Data Cleaning

rename_DPF = data.rename(columns={'DiabetesPedigreeFunction':'DPF'})

columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_fix:
    median_value = data[data[column] != 0][column].median()
    data[column] = data[column].replace(0, median_value)

# Splitting Dataset
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling class imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_balanced, y_train_balanced)

# Model Performance
predictions = model.predict(X_test_scaled)
print("\nModel Performance:")
print("-----------------")
print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
print("\nDetailed Performance Report:")
print(classification_report(y_test, predictions))

# Feature Importance Visualization
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(13, 7))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Health Measurements Impact on Diabetes Prediction')
plt.xlabel('Features')
plt.tight_layout()
plt.show()


################################## Diabetes Prediction System ##########################################



def validate_input(value, feature_name, valid_range):
    """Validate if the input is within the acceptable range"""
    min_val, max_val = map(float, valid_range.split('-'))
    if min_val <= value <= max_val:
        return True
    print(f"Warning: {feature_name} value should be between {min_val} and {max_val}")
    return False

def predict_diabetes():
    features = [
        {"name": "Pregnancies", "description": "Number of pregnancies", "range": "0-17"},
        {"name": "Glucose", "description": "Plasma glucose concentration (mg/dL)", "range": "0-200"},
        {"name": "BloodPressure", "description": "Diastolic blood pressure (mm Hg)", "range": "0-122"},
        {"name": "SkinThickness", "description": "Triceps skin fold thickness (mm)", "range": "0-99"},
        {"name": "Insulin", "description": "2-Hour serum insulin (mu U/ml)", "range": "0-846"},
        {"name": "BMI", "description": "Body mass index", "range": "0-67.1"},
        {"name": "DPF", "description": "Diabetes pedigree function", "range": "0.078-2.42"},
        {"name": "Age", "description": "Age in years", "range": "21-81"}
    ]
    
    patient_data = {}
    print("\n=== Diabetes Risk Assessment ===")
    
    for feature in features:
        while True:
            print(f"\n{feature['name']}")
            print(f"Description: {feature['description']}")
            print(f"Valid range: {feature['range']}")
            
            try:
                value = float(input("Enter value: "))
                if validate_input(value, feature['name'], feature['range']):
                    patient_data[feature['name']] = value
                    break
            except ValueError:
                print("Please enter a valid number")

    # DF with the same features as training data
    patient_df = pd.DataFrame([patient_data])
    patient_df = patient_df[X.columns]
    
    # Scale the data
    patient_scaled = scaler.transform(patient_df)
    
    # Prediction
    prediction = model.predict(patient_scaled)[0]
    risk_probability = model.predict_proba(patient_scaled)[0][1] * 100
    
    print("\n=== Prediction Results ===")
    print(f"Risk of Diabetes: {risk_probability:.1f}%")
    print("Diagnosis:", "High risk of diabetes" if prediction == 1 else "Low risk of diabetes")
    
    if risk_probability > 70:
        print("Recommendation: Please consult a healthcare provider as soon as possible.")
    elif risk_probability > 30:
        print("Recommendation: Consider scheduling a check-up with your healthcare provider.")
    else:
        print("Recommendation: Continue maintaining a healthy lifestyle.")

# Main execution

print("\n===========================================================================\n")
print('\033[92m' + "\033[1mWELCOME TO DIABETES RISK ASSESSMENT TOOL!\033[0m")
while True:
    response = input("\nWould you like to make a diabetes prediction? (yes/no): ").lower()
    if response == 'yes':
        predict_diabetes()
    else:
        print("\nAffirmative!!!")
        break
    if input("\nWould you like to make another prediction? (yes/no): ").lower() != 'yes':
        break

print("\nThank you for using the Diabetes Risk Assessment tool!")