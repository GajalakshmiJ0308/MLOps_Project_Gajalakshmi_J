# src/api.py

from fastapi import FastAPI
import pandas as pd
from pathlib import Path
from src import analysis

app = FastAPI(
    title="Retail Insights API (Simplified)",
    description="A straightforward API for retail analytics."
)

# --- Data Loading ---
DATA_FILE_PATH = Path(__file__).parent.parent / 'data' / 'processed_customer_data.parquet'
try:
    df = pd.read_parquet(DATA_FILE_PATH)
    rfm_df = analysis.perform_rfm_kmeans_segmentation(df)
    print("✅ Data loaded and initial analysis complete.")
except FileNotFoundError:
    print(f"❌ Error: Processed data file not found at '{DATA_FILE_PATH}'")
    df = pd.DataFrame()
    rfm_df = pd.DataFrame()

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Simplified Retail Insights API"}

@app.get("/performance/stores")
def get_store_performance():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.calculate_store_performance(df).to_dict('index')

@app.get("/customers/top-customers")
def get_top_customers():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.get_top_customers(df).to_dict('records')

@app.get("/customers/value-segmentation")
def get_value_segmentation():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.get_customer_value_segmentation(df).to_dict('records')

@app.get("/insights/discount-impact")
def get_discount_impact():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.get_discount_impact(df).to_dict('records')

@app.get("/insights/seasonality")
def get_seasonality():
    if df.empty: return {"error": "Data not loaded."}
    monthly, quarterly = analysis.analyze_seasonality(df)
    return {
        "monthly": monthly.to_dict('records'),
        "quarterly": quarterly.to_dict('records')
    }

@app.get("/insights/payment-methods")
def get_payment_methods():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.analyze_payment_methods(df).to_dict('records')

@app.get("/customers/rfm-segments")
def get_rfm_segments():
    if rfm_df.empty: return {"error": "Data not loaded."}
    return rfm_df.to_dict('records')

# --- NEW ENDPOINTS ---

@app.get("/customers/repeat-vs-onetime")
def get_repeat_vs_onetime():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.analyze_repeat_vs_onetime_customers(df).to_dict('records')

@app.get("/insights/category-by-segment")
def get_category_by_segment():
    if df.empty or rfm_df.empty: return {"error": "Data not loaded."}
    return analysis.get_category_insights_by_segment(df, rfm_df).to_dict('records')

@app.get("/simulations/campaign")
def simulate_campaign(target_segment: str, discount_pct: float = 0.10):
    if rfm_df.empty: return {"error": "Data not loaded."}
    return analysis.run_campaign_simulation(rfm_df, target_segment, discount_pct)