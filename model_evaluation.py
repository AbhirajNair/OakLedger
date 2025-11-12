import os
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, RandomizedSearchCV

# Suppress XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    f1_score, roc_auc_score, accuracy_score, 
    classification_report, make_scorer
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import randint, uniform

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class ModelEvaluator:
    def __init__(self):
        self.models_dir = "models/"
        self.xgboost_dir = os.path.join(self.models_dir, "xgboost")
        self.results = []
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("📊 Loading and preparing data...")
        df = pd.read_csv("data/financial_features_large.csv")
        
        # Prepare features and target
        features = [
            'age', 'family_size', 'total_income', 'debt_ratio', 
            'expense_ratio', 'savings_ratio', 'credit_score'
        ]
        
        # Convert age to bins
        def convert_age(age):
            if pd.isna(age):
                return 3
            if age < 30:
                return 1
            elif age < 50:
                return 2
            return 3
            
        X = df[features].copy()
        if 'age' in X.columns:
            X['age'] = X['age'].apply(convert_age)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Regression target
        if 'total_expenses' in df.columns:
            y_reg = df['total_expenses']
        else:
            y_reg = None
            
        # Classification target
        if 'financial_health' in df.columns:
            self.label_encoder = LabelEncoder()
            y_clf = self.label_encoder.fit_transform(df['financial_health'])
        else:
            y_clf = None
            
        return X, y_reg, y_clf
    
    def evaluate_model(self, model, X, y, model_name, task_type='regression'):
        """Evaluate a single model using cross-validation."""
        print(f"\n🔍 Evaluating {model_name} ({task_type})...")
        
        # Define scoring metrics based on task type
        if task_type == 'regression':
            scoring = {
                'neg_mse': 'neg_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error',
                'r2': 'r2'
            }
            refit = 'neg_mse'
        else:  # classification
            scoring = {
                'f1_weighted': 'f1_weighted',
                'accuracy': 'accuracy',
                'roc_auc_ovr': 'roc_auc_ovr'
            }
            refit = 'f1_weighted'
        
        # Perform cross-validation
        cv_scores = cross_validate(
            model, X, y, cv=5, scoring=scoring, 
            return_train_score=True, n_jobs=-1
        )
        
        # Calculate metrics
        metrics = {}
        for metric in scoring.keys():
            train_metric = f'train_{metric}'
            test_metric = f'test_{metric}'
            
            if metric.startswith('neg_'):
                # Convert back to positive for negative metrics
                metrics[train_metric] = -np.mean(cv_scores[train_metric])
                metrics[test_metric] = -np.mean(cv_scores[test_metric])
            else:
                metrics[train_metric] = np.mean(cv_scores[train_metric])
                metrics[test_metric] = np.mean(cv_scores[test_metric])
        
        # Store results
        result = {
            'model_name': model_name,
            'task_type': task_type,
            **metrics,
            'fit_time': np.mean(cv_scores['fit_time']),
            'score_time': np.mean(cv_scores['score_time'])
        }
        
        self.results.append(result)
        return result
    
    def hyperparameter_tuning(self, model, param_grid, X, y, model_name, task_type='regression', search_type='grid'):
        """Perform hyperparameter tuning using grid or random search."""
        print(f"\n🎯 Tuning hyperparameters for {model_name} using {search_type} search...")
        
        # Define scoring based on task type
        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'f1_weighted'
        
        # Select search type
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=5, scoring=scoring,
                n_jobs=-1, verbose=1, refit=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_distributions=param_grid, n_iter=10,
                cv=5, scoring=scoring, n_jobs=-1, 
                verbose=1, random_state=RANDOM_STATE, refit=True
            )
        
        # Perform search
        start_time = time.time()
        search.fit(X, y)
        end_time = time.time()
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")
        print(f"Tuning completed in {end_time - start_time:.2f} seconds")
        
        return search.best_estimator_
    
    def plot_results(self):
        """Plot the evaluation results."""
        if not self.results:
            print("No results to plot!")
            return
            
        df = pd.DataFrame(self.results)
        
        # Plot regression metrics
        reg_results = df[df['task_type'] == 'regression']
        if not reg_results.empty:
            self._plot_metric_comparison(
                reg_results, 
                ['test_neg_mse', 'test_neg_mae', 'test_r2'],
                'Regression Model Comparison'
            )
        
        # Plot classification metrics
        clf_results = df[df['task_type'] == 'classification']
        if not clf_results.empty:
            self._plot_metric_comparison(
                clf_results,
                ['test_f1_weighted', 'test_accuracy', 'test_roc_auc_ovr'],
                'Classification Model Comparison'
            )
    
    def _plot_metric_comparison(self, results, metrics, title):
        """Helper function to plot metric comparisons."""
        plt.figure(figsize=(12, 6))
        results = results.set_index('model_name')
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i + 1)
            results[metric].sort_values().plot(kind='barh')
            plt.title(metric.replace('_', ' ').title())
            plt.tight_layout()
        
        plt.suptitle(title, y=1.05)
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('static/images/plots', exist_ok=True)
        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(f'static/images/plots/{filename}', bbox_inches='tight')
        plt.close()
        print(f"\n📊 Plot saved as 'static/images/plots/{filename}'")

def main():
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load and prepare data
    X, y_reg, y_clf = evaluator.load_data()
    
    # Define models to evaluate
    models = {
        'regression': {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'XGBoost Regressor': XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='rmse')
        },
        'classification': {
            'Random Forest Classifier': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'XGBoost Classifier': XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
        }
    }
    
    # Define hyperparameter grids for tuning
    param_grids = {
        'Random Forest Regressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost Regressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'Random Forest Classifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': [None, 'balanced']
        },
        'XGBoost Classifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }
    
    # Evaluate regression models
    if y_reg is not None:
        print("\n" + "="*50)
        print("REGRESSION MODELS EVALUATION")
        print("="*50)
        
        for name, model in models['regression'].items():
            # Evaluate with default parameters
            print(f"\n{'-'*20} {name} {'-'*20}")
            evaluator.evaluate_model(model, X, y_reg, name, 'regression')
            
            # Hyperparameter tuning if grid is defined
            if name in param_grids:
                best_model = evaluator.hyperparameter_tuning(
                    model, param_grids[name], X, y_reg, 
                    f"{name} (Tuned)", 'regression', search_type='random'
                )
                # Evaluate tuned model
                evaluator.evaluate_model(best_model, X, y_reg, f"{name} (Tuned)", 'regression')
    
    # Evaluate classification models
    if y_clf is not None:
        print("\n" + "="*50)
        print("CLASSIFICATION MODELS EVALUATION")
        print("="*50)
        
        for name, model in models['classification'].items():
            print(f"\n{'-'*20} {name} {'-'*20}")
            evaluator.evaluate_model(model, X, y_clf, name, 'classification')
            
            # Hyperparameter tuning if grid is defined
            if name in param_grids:
                best_model = evaluator.hyperparameter_tuning(
                    model, param_grids[name], X, y_clf,
                    f"{name} (Tuned)", 'classification', search_type='random'
                )
                # Evaluate tuned model
                evaluator.evaluate_model(best_model, X, y_clf, f"{name} (Tuned)", 'classification')
    
    # Plot and save results
    evaluator.plot_results()
    
    # Print summary of results
    results_df = pd.DataFrame(evaluator.results)
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    if not results_df.empty:
        # For regression
        if 'test_neg_mse' in results_df.columns:
            print("\nBest Regression Model (by RMSE):")
            best_reg = results_df[results_df['task_type'] == 'regression'].sort_values('test_neg_mse').iloc[0]
            print(f"Model: {best_reg['model_name']}")
            print(f"RMSE: {np.sqrt(best_reg['test_neg_mse']):.4f}")
            print(f"MAE: {best_reg['test_neg_mae']:.4f}")
            print(f"R²: {best_reg['test_r2']:.4f}")
        
        # For classification
        if 'test_f1_weighted' in results_df.columns:
            print("\nBest Classification Model (by F1-score):")
            best_clf = results_df[results_df['task_type'] == 'classification'].sort_values('test_f1_weighted', ascending=False).iloc[0]
            print(f"Model: {best_clf['model_name']}")
            print(f"F1-score: {best_clf['test_f1_weighted']:.4f}")
            print(f"Accuracy: {best_clf['test_accuracy']:.4f}")
            print(f"ROC-AUC: {best_clf['test_roc_auc_ovr']:.4f}")
        
        # Save results to CSV
        results_df.to_csv('model_evaluation_results.csv', index=False)
        print("\n📝 Detailed results saved to 'model_evaluation_results.csv'")
    else:
        print("No results to display!")

if __name__ == "__main__":
    main()
