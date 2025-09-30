# src/analytics_engine.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class AnalyticsEngine:
    """
    An engine to handle all retail data analytics, including
    performance metrics and K-Means based customer segmentation.
    """
    def __init__(self, data_path: str):
        try:
            self.df = pd.read_parquet(data_path)
            # Pre-calculate RFM and clusters on initialization
            self.rfm_df = self._calculate_rfm()
            self.rfm_df_clustered = self._perform_kmeans_segmentation()
            print("✅ AnalyticsEngine initialized and data loaded successfully.")
        except FileNotFoundError:
            print(f"❌ Error: Processed data file not found at '{data_path}'")
            self.df = pd.DataFrame()
            self.rfm_df_clustered = pd.DataFrame()

    def _calculate_rfm(self) -> pd.DataFrame:
        """Calculates Recency, Frequency, and Monetary values for each customer."""
        today = self.df['invoice_date'].max() + pd.Timedelta(days=1)
        
        rfm = self.df.groupby('customer_id').agg({
            'invoice_date': lambda date: (today - date.max()).days,
            'invoice_no': 'nunique',
            'net_sales': 'sum'
        }).rename(columns={
            'invoice_date': 'recency', 
            'invoice_no': 'frequency', 
            'net_sales': 'monetary'
        })
        return rfm

    def _perform_kmeans_segmentation(self, n_clusters: int = 5) -> pd.DataFrame:
        """Performs K-Means clustering on RFM data."""
        if self.rfm_df.empty:
            return pd.DataFrame()

        # Handle potential skewness in data
        rfm_log = self.rfm_df.copy()
        for col in ['recency', 'frequency', 'monetary']:
             # Add a small constant to avoid log(0)
            rfm_log[col] = np.log(rfm_log[col] + 1)

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        
        clustered_data = self.rfm_df.copy()
        clustered_data['cluster'] = kmeans.labels_
        return clustered_data.reset_index()

    def get_store_performance(self) -> pd.DataFrame:
        """Analyzes total sales and transaction count across stores."""
        return self.df.groupby('shopping_mall').agg(
            total_net_sales=('net_sales', 'sum'),
            average_sale_value=('net_sales', 'mean'),
            transaction_count=('invoice_no', 'nunique')
        ).sort_values(by='total_net_sales', ascending=False)

    def get_payment_methods_distribution(self) -> pd.DataFrame:
        """Calculates the distribution of payment methods."""
        return self.df['payment_method'].value_counts(normalize=True).reset_index()

    def get_monthly_sales_trend(self) -> pd.DataFrame:
        """Analyzes monthly sales trends."""
        monthly_sales = self.df.set_index('invoice_date')['net_sales'].resample('M').sum()
        return monthly_sales.reset_index()

    def get_kmeans_cluster_profiles(self) -> pd.DataFrame:
        """Returns the average RFM values for each customer cluster."""
        if self.rfm_df_clustered.empty:
            return pd.DataFrame()
        return self.rfm_df_clustered.groupby('cluster')[['recency', 'frequency', 'monetary']].mean().reset_index()

    def get_category_insights_by_cluster(self) -> pd.DataFrame:
        """Analyzes category popularity across different K-Means clusters."""
        merged_data = pd.merge(self.df, self.rfm_df_clustered[['customer_id', 'cluster']], on='customer_id')
        return merged_data.groupby(['cluster', 'category'])['net_sales'].sum().reset_index()

    def run_campaign_simulation(self, target_cluster: int, discount: float = 0.1) -> dict:
        """Projects the ROI of a marketing campaign targeting a specific customer cluster."""
        target_customers = self.rfm_df_clustered[self.rfm_df_clustered['cluster'] == target_cluster]
        
        if target_customers.empty:
            return {"error": "No customers found in the selected cluster."}
            
        customer_count = len(target_customers)
        # Use the average monetary value of the cluster for simulation
        avg_spend = target_customers['monetary'].mean()
        
        # Assume campaign generates sales equal to historical average with a 15% uplift
        uplift_factor = 1.15 
        projected_revenue = (avg_spend * customer_count) * uplift_factor
        campaign_cost = projected_revenue * discount
        net_profit = projected_revenue - campaign_cost
        projected_roi = (net_profit - campaign_cost) / campaign_cost if campaign_cost > 0 else 0
        
        return {
            "target_cluster": target_cluster,
            "customer_count": customer_count,
            "projected_revenue": f"${projected_revenue:,.2f}",
            "campaign_cost": f"${campaign_cost:,.2f}",
            "projected_roi_percent": f"{projected_roi * 100:.2f}%"
        }
