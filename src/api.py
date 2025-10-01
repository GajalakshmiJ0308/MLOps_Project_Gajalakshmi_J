# src/api.py

from fastapi import FastAPI, HTTPException
from pathlib import Path
from src.analytics_engine import AnalyticsEngine

app = FastAPI(
    title="Retail Insights API",
    description="An API for retail analytics powered by an AnalyticsEngine and K-Means clustering."
)

# Define path to the data and instantiate the engine
DATA_FILE_PATH = Path(__file__).parent.parent / 'data' / 'processed_customer_data.parquet'
engine = AnalyticsEngine(str(DATA_FILE_PATH))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Retail Insights API. To access the API Main Page, include '/docs' at the end of the url."}

# Endpoints
@app.get("/performance/stores")
def get_store_performance():
    if engine.df.empty:
        raise HTTPException(status_code=404, detail="Data not loaded.")
    return engine.get_store_performance().to_dict('index')

@app.get("/insights/payment-methods")
def get_payment_methods():
    if engine.df.empty:
        raise HTTPException(status_code=404, detail="Data not loaded.")
    return engine.get_payment_methods_distribution().to_dict('records')

@app.get("/trends/monthly-sales")
def get_monthly_sales():
    if engine.df.empty:
        raise HTTPException(status_code=404, detail="Data not loaded.")
    return engine.get_monthly_sales_trend().to_dict('records')

@app.get("/segmentation/kmeans-clusters")
def get_kmeans_clusters():
    if engine.rfm_df_clustered.empty:
        raise HTTPException(status_code=404, detail="Segmentation data not available.")
    return engine.rfm_df_clustered.to_dict('records')

@app.get("/segmentation/cluster-profiles")
def get_cluster_profiles():
    if engine.rfm_df_clustered.empty:
        raise HTTPException(status_code=404, detail="Segmentation data not available.")
    return engine.get_kmeans_cluster_profiles().to_dict('records')

@app.get("/insights/category-by-cluster")
def get_category_by_cluster():
    if engine.df.empty or engine.rfm_df_clustered.empty:
        raise HTTPException(status_code=404, detail="Data not available.")
    return engine.get_category_insights_by_cluster().to_dict('records')

@app.get("/simulations/campaign")
def run_campaign_simulation(target_cluster: int, discount: float = 0.1):
    if engine.rfm_df_clustered.empty:
        raise HTTPException(status_code=404, detail="Data not available.")
    return engine.run_campaign_simulation(target_cluster, discount)

