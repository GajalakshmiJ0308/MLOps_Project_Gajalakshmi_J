# src/data_processing.py

import pandas as pd
import numpy as np
import os
from pathlib import Path

def process_data(input_filepath: str, output_filepath: str):
    """
    Loads, cleans, and engineers features for the retail dataset.
    This version includes handling for missing values.
    """
    print("🚀 Starting data processing...")
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"❌ Error: Input data file not found at '{input_filepath}'")
        print("👉 Please ensure 'customer_shopping_data.csv' is in the 'data' folder.")
        return

    # 1. Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # 2. Handle missing values
    if df.isnull().sum().any():
        print(f"🔍 Found {df.isnull().sum().sum()} missing values. Dropping rows with NaNs.")
        df.dropna(inplace=True)

    # 3. Convert data types
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)

    # 4. Feature Engineering
    df['total_sales'] = df['quantity'] * df['price']
    
    # Simulate discounts for profitability analysis
    np.random.seed(42)
    df['discount_percentage'] = np.random.uniform(0.02, 0.15, df.shape[0])
    df['discount_amount'] = df['total_sales'] * df['discount_percentage']
    df['net_sales'] = df['total_sales'] - df['discount_amount']
    
    # Date-based features
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['day_of_week'] = df['invoice_date'].dt.day_name()
    df['hour'] = df['invoice_date'].dt.hour

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # 5. Save processed data to a more efficient format
    df.to_parquet(output_filepath, index=False)
    print(f"✅ Data processing complete. Processed file saved to {output_filepath}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    INPUT_PATH = project_root / 'data' / 'customer_shopping_data.csv'
    OUTPUT_PATH = project_root / 'data' / 'processed_customer_data.parquet'
    
    process_data(str(INPUT_PATH), str(OUTPUT_PATH))
