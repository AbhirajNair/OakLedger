import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_user_data(user_id):
    """Generate synthetic financial data for a single user."""
    # Base financial metrics
    age = random.randint(18, 70)
    family_size = random.choices(
        [1, 2, 3, 4, 5],
        weights=[0.2, 0.3, 0.25, 0.15, 0.1]
    )[0]
    
    # Base income based on age and family size
    base_income = 30000 + (age * 500) + (family_size * 2000)
    
    # Add some randomness
    total_income = max(25000, np.random.normal(base_income, base_income * 0.3))
    
    # Generate expenses (60-90% of income)
    expense_ratio = np.random.uniform(0.6, 0.9)
    total_expenses = total_income * expense_ratio
    
    # Calculate savings
    savings = max(0, total_income - total_expenses)
    savings_ratio = savings / total_income if total_income > 0 else 0
    
    # Generate debt ratio (10-50% of income)
    debt_ratio = np.random.uniform(10, 50)
    
    # Generate credit score (300-850)
    credit_score = int(np.clip(np.random.normal(650, 100), 300, 850))
    
    # Employment sector
    sectors = ['IT', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail', 'Government', 'Other']
    employment_sector = random.choice(sectors)
    
    # Determine financial health based on metrics
    if savings_ratio > 0.2 and debt_ratio < 20 and credit_score > 700:
        financial_health = 'Good'
    elif savings_ratio > 0.1 and debt_ratio < 40 and credit_score > 600:
        financial_health = 'Moderate'
    else:
        financial_health = 'Needs Improvement'
    
    return {
        'user_id': f'user_{user_id:04d}',
        'age': age,
        'family_size': family_size,
        'total_income': round(total_income, 2),
        'total_expenses': round(total_expenses, 2),
        'savings': round(savings, 2),
        'savings_ratio': round(savings_ratio, 3),
        'debt_ratio': round(debt_ratio, 3),
        'expense_ratio': round(expense_ratio, 3),
        'credit_score': credit_score,
        'employment_sector': employment_sector,
        'financial_health': financial_health
    }

def generate_large_dataset(num_users=1000, output_file='data/financial_features_large.csv'):
    """Generate a large dataset of synthetic financial data."""
    print(f"🚀 Generating synthetic data for {num_users} users...")
    
    # Generate user data
    users_data = [generate_user_data(i+1) for i in range(num_users)]
    
    # Create DataFrame
    df = pd.DataFrame(users_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Dataset with {len(df)} users saved to {output_file}")
    print("\nDataset Summary:")
    print(f"- Total Users: {len(df)}")
    print(f"- Average Income: ${df['total_income'].mean():,.2f}")
    print(f"- Average Savings Ratio: {df['savings_ratio'].mean():.1%}")
    print(f"- Financial Health Distribution:")
    print(df['financial_health'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
    return df

if __name__ == "__main__":
    # Generate dataset with 1000 users
    df = generate_large_dataset(num_users=1000)
    
    # Generate monthly data for all users
    print("\n🔄 Generating monthly financial data...")
    months = pd.date_range(start="2024-01-01", periods=12, freq="M")
    synthetic_rows = []
    
    for _, row in df.iterrows():
        for month in months:
            # Add some monthly variation (5-15%)
            income = row['total_income'] * np.random.uniform(0.95, 1.05)
            expenses = row['total_expenses'] * np.random.uniform(0.9, 1.1)
            savings = income - expenses
            savings_ratio = max(savings / income, 0) if income > 0 else 0
            
            synthetic_rows.append({
                'user_id': row['user_id'],
                'month': month.strftime('%Y-%m'),
                'total_income': round(income, 2),
                'total_expenses': round(expenses, 2),
                'savings': round(savings, 2),
                'savings_ratio': round(savings_ratio, 3),
                'debt_ratio': row['debt_ratio'] * np.random.uniform(0.95, 1.05),
                'expense_ratio': expenses / income if income > 0 else 0,
                'financial_health': row['financial_health']
            })
    
    monthly_df = pd.DataFrame(synthetic_rows)
    monthly_file = 'data/monthly_financial_data_large.csv'
    monthly_df.to_csv(monthly_file, index=False)
    print(f"✅ Monthly financial data saved to {monthly_file}")
    print(f"Total records: {len(monthly_df):,}")
    print("\nSample of generated data:")
    print(monthly_df.head())
