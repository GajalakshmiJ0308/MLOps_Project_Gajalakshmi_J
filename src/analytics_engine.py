import pandas as pd
import numpy as np

# 1. Store vs. Region Performance
def calculate_store_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Analyzes total and average sales across stores."""
    return data.groupby('shopping_mall')['net_sales'].agg(['sum', 'mean']).sort_values(by='sum', ascending=False)

# 2. Top Customers
def get_top_customers(data: pd.DataFrame, top_n_percent: float = 0.1) -> pd.DataFrame:
    """Identifies the top N% of customers by total net sales."""
    customer_sales = data.groupby('customer_id')['net_sales'].sum().sort_values(ascending=False)
    top_n_count = int(len(customer_sales) * top_n_percent)
    return customer_sales.head(top_n_count).reset_index()

# 3. High vs. Low-value Segmentation
def get_customer_value_segmentation(data: pd.DataFrame) -> pd.DataFrame:
    """Segments customers into High-Value and Low-Value based on the median of total spend."""
    customer_sales = data.groupby('customer_id')['net_sales'].sum().reset_index()
    median_spend = customer_sales['net_sales'].median()
    customer_sales['value_segment'] = pd.cut(
        customer_sales['net_sales'],
        bins=[-np.inf, median_spend, np.inf],
        labels=['Low-Value', 'High-Value']
    )
    return customer_sales['value_segment'].value_counts().reset_index()

# 4. Discount Impact on Profitability
def get_discount_impact(data: pd.DataFrame) -> pd.DataFrame:
    """Computes total sales, discounts, and net sales per category."""
    return data.groupby('category').agg(
        total_sales=('total_sales', 'sum'),
        total_discount=('discount_amount', 'sum'),
        net_sales=('net_sales', 'sum')
    ).sort_values(by='net_sales', ascending=False).reset_index()

# 5. Seasonality Analysis
def analyze_seasonality(data: pd.DataFrame) -> pd.DataFrame:
    """Analyzes monthly sales trends."""
    monthly_sales = data.set_index('invoice_date').resample('M')['net_sales'].sum()
    return monthly_sales.reset_index()

# 6. Payment Method Preference
def analyze_payment_methods(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates the distribution of payment methods."""
    return data['payment_method'].value_counts(normalize=True).reset_index()

# 7. RFM Analysis
def calculate_rfm(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, Monetary scores and segments customers."""
    today = data['invoice_date'].max() + pd.Timedelta(days=1)
    
    rfm = data.groupby('customer_id').agg({
        'invoice_date': lambda date: (today - date.max()).days,
        'invoice_no': 'nunique',
        'net_sales': 'sum'
    }).rename(columns={
        'invoice_date': 'recency', 'invoice_no': 'frequency', 'net_sales': 'monetary'
    })

    rfm['r_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1])
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['m_score'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4])
    rfm['segment'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

    def map_segment_to_name(s):
        if s in ['444', '434', '443']: return 'Champions'
        elif s in ['344', '433', '442']: return 'Loyal Customers'
        elif s in ['422', '322', '332', '432']: return 'Potential Loyalists'
        elif s in ['111', '112', '121']: return 'Lost Customers'
        else: return 'At Risk'
    
    rfm['segment_name'] = rfm['segment'].apply(map_segment_to_name)
    return rfm.reset_index()

# 8. Repeat Customer vs. One-time
def get_repeat_vs_onetime_customers(data: pd.DataFrame) -> pd.DataFrame:
    """Compares sales contribution from repeat vs. one-time customers."""
    customer_frequency = data.groupby('customer_id')['invoice_no'].nunique().reset_index()
    customer_frequency['customer_type'] = np.where(customer_frequency['invoice_no'] > 1, 'Repeat Customer', 'One-Time Customer')
    
    merged_data = pd.merge(data, customer_frequency[['customer_id', 'customer_type']], on='customer_id')
    
    return merged_data.groupby('customer_type')['net_sales'].sum().reset_index()

# 9. Category-wise Insights
def get_category_insights_by_segment(data: pd.DataFrame, rfm_data: pd.DataFrame) -> pd.DataFrame:
    """Analyzes category popularity across different RFM segments."""
    merged_data = pd.merge(data, rfm_data[['customer_id', 'segment_name']], on='customer_id')
    top_segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk']
    filtered_data = merged_data[merged_data['segment_name'].isin(top_segments)]
    
    return filtered_data.groupby(['segment_name', 'category'])['net_sales'].sum().reset_index()

# 10. Campaign Simulation
def run_campaign_simulation(data: pd.DataFrame, rfm_data: pd.DataFrame, target_segment: str = 'Champions', discount: float = 0.1) -> dict:
    """Projects the ROI of a marketing campaign targeting a specific customer segment."""
    target_customers = rfm_data[rfm_data['segment_name'] == target_segment]['customer_id']
    
    if target_customers.empty:
        return {"projected_revenue": 0, "campaign_cost": 0, "projected_roi": 0, "customer_count": 0}
        
    historical_sales = data[data['customer_id'].isin(target_customers)]
    avg_spend_per_customer = historical_sales.groupby('customer_id')['net_sales'].sum().mean()
    
    # Assume campaign generates sales equal to the segment's historical average spend with an uplift
    uplift_factor = 1.1 # Assume a 10% uplift in sales due to the campaign
    projected_revenue = (avg_spend_per_customer * len(target_customers)) * uplift_factor
    campaign_cost = projected_revenue * discount
    net_profit = projected_revenue - campaign_cost
    projected_roi = (net_profit - campaign_cost) / campaign_cost if campaign_cost > 0 else 0
    
    return {
        "projected_revenue": projected_revenue,
        "campaign_cost": campaign_cost,
        "projected_roi": projected_roi * 100,
        "customer_count": len(target_customers)
    }
