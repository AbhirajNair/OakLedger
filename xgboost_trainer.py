import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import os

def train_xgboost_models():
    print("🚀 Training XGBoost models...")
    
    # Load data
    df = pd.read_csv("data/financial_features_large.csv")
    
    # Prepare features
    # Regression features
    reg_features = ['age', 'family_size', 'total_income', 'debt_ratio', 
                   'expense_ratio', 'savings_ratio', 'credit_score']
    
    # Classification features
    clf_features = ['total_income', 'total_expenses', 'savings', 
                   'savings_ratio', 'debt_ratio', 'expense_ratio']
    
    # Encode target
    label_encoder = joblib.load("models/label_encoder.pkl")
    df['financial_health_encoded'] = label_encoder.transform(df['financial_health'])
    
    # Split data
    X_reg = df[reg_features]
    y_reg = df['total_expenses']
    
    X_clf = df[clf_features]
    y_clf = df['financial_health_encoded']
    
    # Split into train/test
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Train XGBoost Regressor
    print("\n🏗️  Training XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    xgb_reg.fit(X_reg_train, y_reg_train)
    
    # Evaluate regression
    y_pred_reg = xgb_reg.predict(X_reg_test)
    mse = mean_squared_error(y_reg_test, y_pred_reg)
    r2 = 1 - (mse / np.var(y_reg_test))
    print(f"XGBoost Regressor MSE: {mse:.2f}")
    print(f"XGBoost Regressor R²: {r2:.4f}")
    
    # Feature importance
    print("\n🔍 Top 5 Important Features (Regression):")
    feat_imp = pd.Series(xgb_reg.feature_importances_, index=reg_features)
    print(feat_imp.sort_values(ascending=False).head())
    
    # Train XGBoost Classifier
    print("\n🏗️  Training XGBoost Classifier...")
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    xgb_clf.fit(X_clf_train, y_clf_train)
    
    # Evaluate classification
    y_pred_clf = xgb_clf.predict(X_clf_test)
    acc = accuracy_score(y_clf_test, y_pred_clf)
    print(f"\nXGBoost Classifier Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_clf_test, y_pred_clf, target_names=label_encoder.classes_))
    
    # Save models
    os.makedirs("models/xgboost", exist_ok=True)
    joblib.dump(xgb_reg, "models/xgboost/xgboost_regressor.pkl")
    joblib.dump(xgb_clf, "models/xgboost/xgboost_classifier.pkl")
    
    # Save feature importance
    feature_importance = {
        'regression': dict(zip(reg_features, xgb_reg.feature_importances_)),
        'classification': dict(zip(clf_features, xgb_clf.feature_importances_))
    }
    joblib.dump(feature_importance, "models/xgboost/feature_importance.pkl")
    
    print("\n✅ XGBoost models and feature importance saved to models/xgboost/")
    
    return {
        'regression': {
            'mse': float(mse),
            'r2': float(r2),
            'feature_importance': feature_importance['regression']
        },
        'classification': {
            'accuracy': float(acc),
            'feature_importance': feature_importance['classification']
        }
    }

if __name__ == "__main__":
    train_xgboost_models()
