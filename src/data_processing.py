import pandas as pd
import numpy as np
import os
from pathlib import Path

def process_data(input_filepath: str, output_filepath: str):
    """
    Loads, cleans, and engineers features for the retail dataset.
    """
    print("Starting data processing...")
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"❌ Error: Input data file not found at '{input_filepath}'")
        print("👉 Please ensure 'customer_shopping_data.csv' is in the 'data' folder.")
        return

    # 1. Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # 2. Convert invoice_date to datetime
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)

    # 3. Feature Engineering
    df['total_sales'] = df['quantity'] * df['price']
    np.random.seed(42)
    df['discount_percentage'] = np.random.uniform(0.02, 0.15, df.shape[0])
    df['discount_amount'] = df['total_sales'] * df['discount_percentage']
    df['net_sales'] = df['total_sales'] - df['discount_amount']
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['quarter'] = df['invoice_date'].dt.quarter
    df['day_of_week'] = df['invoice_date'].dt.day_name()

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # 4. Save processed data
    df.to_parquet(output_filepath)
    print(f"✅ Data processing complete. Processed file saved to {output_filepath}")

if __name__ == "__main__":
    # Build a reliable path from this script's location to the project root and then to the data folder.
    project_root = Path(__file__).parent.parent
    INPUT_PATH = project_root / 'data' / 'customer_shopping_data.csv'
    OUTPUT_PATH = project_root / 'data' / 'processed_customer_data.parquet'
    
    process_data(INPUT_PATH, OUTPUT_PATH)