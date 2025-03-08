import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
num_samples = 10000
data = {
    "Customer_Age": np.random.randint(25, 65, num_samples),
    "Income": np.random.uniform(30000, 200000, num_samples),
    "Property_Value": np.random.uniform(100000, 1000000, num_samples),
    "Loan_Amount": np.random.uniform(50000, 900000, num_samples),
    "Loan_Term": np.random.choice([10, 15, 20, 25, 30], num_samples),
    "Interest_Rate": np.random.uniform(2.5, 10.5, num_samples),
    "Previous_Advance_Payments": np.random.randint(0, 5, num_samples),
    "Credit_Score": np.random.randint(550, 850, num_samples),
    "Employment_Status": np.random.choice(["Employed", "Self-Employed", "Unemployed"], num_samples),
    "Advance_Payment": np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # 30% advance payments
}

# Convert to DataFrame
df_real_estate = pd.DataFrame(data)

# Encode categorical variables
df_real_estate = pd.get_dummies(df_real_estate, columns=["Employment_Status"], drop_first=True)

# Display first few rows
print(df_real_estate.head())

# Split data into features and target
X = df_real_estate.drop(columns=["Advance_Payment"])
y = df_real_estate["Advance_Payment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
print(f"ROC-AUC Score: {roc_auc:.4f}")
