import pandas as pd
import numpy as np

def debug_sample_predictions():
    # Load the data
    df = pd.read_csv("data/final_predictions.csv")
    
    print("DataFrame shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Try to create sample data
    n_samples = 10
    sample_df = df.sample(min(n_samples, len(df)))
    
    print("\nSample DataFrame shape:", sample_df.shape)
    
    # Check for missing values
    print("\nMissing values in sample:")
    print(sample_df.isnull().sum())
    
    # Print the data we'll use for the table
    print("\nTable data:")
    table_data = []
    for _, row in sample_df.iterrows():
        row_data = [
            row.get('user_id', 'N/A'),
            f"${row.get('total_income', 0):,.2f}",
            f"${row.get('total_expenses', 0):,.2f}",
            f"${row.get('savings', 0):,.2f}",
            row.get('financial_health', 'N/A'),
            row.get('predicted_financial_health', 'N/A'),
            '✓' if row.get('financial_health') == row.get('predicted_financial_health') else '✗'
        ]
        print(row_data)
        table_data.append(row_data)
    
    print("\nTable data length:", len(table_data))
    
    # Check the data types
    print("\nData types:")
    for col in sample_df.columns:
        print(f"{col}: {sample_df[col].dtype}")

if __name__ == "__main__":
    debug_sample_predictions()
