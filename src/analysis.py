# src/analysis.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Case Scenario Functions ---

# 1. Store Performance
def calculate_store_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes sales volume and revenue across stores."""
    return df.groupby('shopping_mall')['net_sales'].agg(['sum', 'mean', 'count']).sort_values(by='sum', ascending=False)

# 2. Top Customers
def get_top_customers(df: pd.DataFrame, percentile: float = 0.90) -> pd.DataFrame:
    """Identifies top customers by purchase value based on a given percentile."""
    customer_sales = df.groupby('customer_id')['net_sales'].sum()
    top_tier_threshold = customer_sales.quantile(percentile)
    top_customers = customer_sales[customer_sales >= top_tier_threshold]
    return top_customers.reset_index().sort_values(by='net_sales', ascending=False)

# 3. High vs. Low-Value Segmentation
def get_customer_value_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Classifies customers into High-Value and Low-Value segments."""
    customer_sales = df.groupby('customer_id')['net_sales'].sum()
    median_spend = customer_sales.median()
    segments = pd.cut(customer_sales, bins=[-np.inf, median_spend, np.inf], labels=['Low-Value', 'High-Value'])
    return segments.value_counts().reset_index()

# 4. Discount Impact on Profitability
def get_discount_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Computes total sales, discounts, and net sales per product category."""
    return df.groupby('category').agg(
        total_sales=('total_sales', 'sum'),
        total_discount=('discount_amount', 'sum'),
        net_sales=('net_sales', 'sum')
    ).sort_values(by='net_sales', ascending=False).reset_index()

# 5. Seasonality Analysis
def analyze_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes monthly and quarterly sales trends."""
    monthly_sales = df.set_index('invoice_date')['net_sales'].resample('ME').sum().reset_index()
    quarterly_sales = df.set_index('invoice_date')['net_sales'].resample('QE').sum().reset_index()
    return monthly_sales, quarterly_sales

# 6. Payment Method Preference
def analyze_payment_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the distribution of payment methods."""
    return df['payment_method'].value_counts(normalize=True).reset_index()

# 7 & 8 (ML Model). RFM Analysis with K-Means Clustering
def perform_rfm_kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Performs RFM analysis and then uses K-Means to segment customers."""
    today = df['invoice_date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda date: (today - date.max()).days,
        'invoice_no': 'nunique',
        'net_sales': 'sum'
    }).rename(columns={'invoice_date': 'recency', 'invoice_no': 'frequency', 'net_sales': 'monetary'})

    rfm_log = np.log1p(rfm)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_profiles = rfm.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
    ordered_clusters = cluster_profiles.sort_values(by='monetary', ascending=False).index
    
    segment_map = {ordered_clusters[0]: 'Champions', ordered_clusters[1]: 'Loyal', ordered_clusters[2]: 'At Risk', ordered_clusters[3]: 'Needs Attention'}
    
    rfm['segment'] = rfm['cluster'].map(segment_map)
    return rfm.reset_index()

# --- NEW FUNCTIONS FOR SCENARIOS 8, 9, 10 ---

# 8. Repeat Customer vs. One-time
def analyze_repeat_vs_onetime_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Compares sales contribution and count of repeat vs. one-time customers."""
    customer_frequency = df.groupby('customer_id')['invoice_no'].nunique().reset_index()
    customer_frequency['customer_type'] = np.where(customer_frequency['invoice_no'] > 1, 'Repeat', 'One-Time')
    
    merged_data = pd.merge(df, customer_frequency[['customer_id', 'customer_type']], on='customer_id')
    
    summary = merged_data.groupby('customer_type').agg(
        total_sales=('net_sales', 'sum'),
        customer_count=('customer_id', 'nunique')
    ).reset_index()
    return summary

# 9. Category-wise Insights
def get_category_insights_by_segment(df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes which product categories are popular within each customer segment."""
    merged_data = pd.merge(df, rfm_df[['customer_id', 'segment']], on='customer_id')
    category_insights = merged_data.groupby(['segment', 'category'])['net_sales'].sum().reset_index()
    return category_insights.sort_values(by=['segment', 'net_sales'], ascending=[True, False])

# 10. Campaign Simulation
def run_campaign_simulation(rfm_df: pd.DataFrame, target_segment: str, discount_pct: float = 0.10) -> dict:
    """Models the potential ROI for a marketing campaign targeting a specific segment."""
    target_customers = rfm_df[rfm_df['segment'] == target_segment]
    
    if target_customers.empty:
        return {"error": f"No customers found in segment: {target_segment}"}
        
    customer_count = len(target_customers)
    avg_spend = target_customers['monetary'].mean()
    
    # Assume a 15% uplift in sales from the campaign
    uplift_factor = 1.15
    projected_revenue = customer_count * avg_spend * uplift_factor
    campaign_cost = projected_revenue * discount_pct
    
    # ROI = (Net Profit / Investment) * 100
    # Net Profit = Projected Revenue - Campaign Cost
    # Investment = Campaign Cost
    net_profit = projected_revenue - campaign_cost
    roi = (net_profit / campaign_cost) * 100 if campaign_cost > 0 else 0
    
    return {
        "target_segment": target_segment,
        "targeted_customers": customer_count,
        "projected_revenue": projected_revenue,
        "campaign_cost": campaign_cost,
        "projected_roi_percent": roi
    }