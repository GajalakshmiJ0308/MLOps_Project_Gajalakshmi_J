from fastapi import FastAPI
import pandas as pd
from src import analysis
from pathlib import Path

app = FastAPI(
    title="Retail Insights API",
    description="API for retail store performance, customer segmentation, and sales insights."
)

DATA_FILE_PATH = Path(__file__).parent.parent / 'data' / 'processed_customer_data.parquet'

try:
    df = pd.read_parquet(DATA_FILE_PATH)
    rfm_df = analysis.calculate_rfm(df)
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Processed data file not found at '{DATA_FILE_PATH}'")
    df = pd.DataFrame()
    rfm_df = pd.DataFrame()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Retail Insights API"}

# Existing endpoints
@app.get("/performance/stores")
def get_store_performance():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.calculate_store_performance(df).to_dict('index')

@app.get("/insights/payment-methods")
def get_payment_methods():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.analyze_payment_methods(df).to_dict('records')

@app.get("/customers/rfm-segments")
def get_rfm_segments():
    if rfm_df.empty: return {"error": "Data not loaded."}
    return rfm_df.to_dict('records')

@app.get("/insights/seasonality")
def get_seasonality():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.analyze_seasonality(df).to_dict('records')

@app.get("/customers/top-customers")
def get_top_customers_endpoint():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.get_top_customers(df).to_dict('records')

@app.get("/customers/value-segmentation")
def get_value_segmentation_endpoint():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.get_customer_value_segmentation(df).to_dict('records')

@app.get("/insights/discount-impact")
def get_discount_impact_endpoint():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.get_discount_impact(df).to_dict('records')

@app.get("/customers/repeat-vs-onetime")
def get_repeat_vs_onetime_endpoint():
    if df.empty: return {"error": "Data not loaded."}
    return analysis.get_repeat_vs_onetime_customers(df).to_dict('records')

@app.get("/insights/category-by-segment")
def get_category_by_segment_endpoint():
    if df.empty or rfm_df.empty: return {"error": "Data not loaded."}
    return analysis.get_category_insights_by_segment(df, rfm_df).to_dict('records')

@app.get("/simulations/campaign")
def run_campaign_simulation_endpoint(target_segment: str = 'Champions', discount: float = 0.1):
    if df.empty or rfm_df.empty: return {"error": "Data not loaded."}
    return analysis.run_campaign_simulation(df, rfm_df, target_segment, discount)
