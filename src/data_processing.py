# src/data_processing.py

import pandas as pd
import numpy as np
import os
from pathlib import Path

def process_data(input_filepath: str, output_filepath: str):
    """
    Loads raw data, cleans it, engineers new features, and saves the result.
    This script forms the first step in our data pipeline.
    """
    print("Starting data processing...")
    
    # Load data
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"❌ Error: Input data file not found at '{input_filepath}'")
        return

    # Clean column names (e.g., 'Shopping Mall' -> 'shopping_mall')
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Convert invoice_date to a proper datetime format
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)
    
    # Handle any potential missing values by removing them
    df.dropna(inplace=True)

    # --- Feature Engineering ---
    # Calculate total sales before discounts
    df['total_sales'] = df['quantity'] * df['price']
    
    # Simulate a discount feature for profitability analysis
    np.random.seed(42) # for reproducibility
    df['discount_percentage'] = np.random.uniform(0.02, 0.15, df.shape[0])
    df['discount_amount'] = df['total_sales'] * df['discount_percentage']
    
    # Calculate final sales after discount
    df['net_sales'] = df['total_sales'] - df['discount_amount']
    
    # Extract time-based features for trend analysis
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['quarter'] = df['invoice_date'].dt.quarter

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Save the processed data to a Parquet file for efficiency
    df.to_parquet(output_filepath, index=False)
    print(f"✅ Data processing complete. Output saved to {output_filepath}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    INPUT_PATH = project_root / 'data' / 'customer_shopping_data.csv'
    OUTPUT_PATH = project_root / 'data' / 'processed_customer_data.parquet'
    process_data(str(INPUT_PATH), str(OUTPUT_PATH))