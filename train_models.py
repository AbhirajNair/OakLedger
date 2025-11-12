import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

print("📊 Starting model training with the new dataset...")

# -----------------------------
# 1. Load and prepare data
# -----------------------------
print("\n📂 Loading data...")
df = pd.read_csv("data/financial_features_large.csv")
print(f"✅ Data loaded. Shape: {df.shape}")

# Prepare features
regression_features = ['age', 'family_size', 'total_income', 'debt_ratio', 'expense_ratio', 'savings_ratio', 'credit_score']
classification_features = ['total_income', 'total_expenses', 'savings', 'savings_ratio', 'debt_ratio', 'expense_ratio']

# Handle missing values
df = df.dropna(subset=regression_features + classification_features + ['financial_health'])

# Encode the target variable
label_encoder = LabelEncoder()
df['financial_health_encoded'] = label_encoder.fit_transform(df['financial_health'])

# Split into features and targets
X_reg = df[regression_features]
y_reg = df['total_expenses']

X_clf = df[classification_features]
y_clf = df['financial_health_encoded']

# Split into train and test sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Scale features
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

scaler_clf = StandardScaler()
X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
X_clf_test_scaled = scaler_clf.transform(X_clf_test)

# -----------------------------
# 2. Train Regression Model
# -----------------------------
print("\n🏗️  Training Regression Model...")
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg_train_scaled, y_reg_train)

# Evaluate regression
y_reg_pred = reg_model.predict(X_reg_test_scaled)
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"\n📈 Regression Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# -----------------------------
# 3. Train Classification Model
# -----------------------------
print("\n🏗️  Training Classification Model...")
clf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf_model.fit(X_clf_train_scaled, y_clf_train)

# Evaluate classification
y_clf_pred = clf_model.predict(X_clf_test_scaled)
print("\n📊 Classification Report:")
print(classification_report(
    y_clf_test, 
    y_clf_pred, 
    target_names=label_encoder.classes_
))

# -----------------------------
# 4. Save Models and Artifacts
# -----------------------------
print("\n💾 Saving models and artifacts...")
os.makedirs("models", exist_ok=True)

# Save models
joblib.dump(reg_model, "models/linear_regression.pkl")
joblib.dump(clf_model, "models/financial_health_model.pkl")
joblib.dump(scaler_reg, "models/reg_scaler.pkl")
joblib.dump(scaler_clf, "models/clf_scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("✅ Models and artifacts saved to 'models/' directory.")
print("\n🎉 Model training complete! You can now use predict_pipeline.py with the updated models.")
