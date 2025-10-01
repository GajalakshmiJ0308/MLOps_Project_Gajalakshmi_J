# src/dashboard.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page & API Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Advanced Retail Insights Dashboard",
    page_icon="💡"
)
# Use the Render URL from your deployment
API_BASE_URL = "https://retail-insights-api-gajalakshmi-j.onrender.com" # Replace if you deploy a new one

# --- Helper Function ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_data(endpoint: str, params: dict = None):
    """Fetches data from the FastAPI endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}. Is the API server running at {API_BASE_URL}?")
        return None

# --- Main Dashboard UI ---
st.title("💡 Advanced Retail & Customer Insights")
st.markdown("An interactive dashboard powered by an ML-driven analytics API.")

# --- Tabbed Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Performance Overview", 
    "👥 Customer Segmentation (K-Means)", 
    "📊 Category & Trends", 
    "🎯 Campaign Simulator"
])

# --- Tab 1: Performance Overview ---
with tab1:
    st.header("Overall Business Performance")
    
    col1, col2, col3 = st.columns(3)
    # Using cluster data to get customer count and sales
    cluster_data = fetch_data("segmentation/kmeans-clusters")
    if cluster_data:
        df = pd.DataFrame(cluster_data)
        total_customers = df['customer_id'].nunique()
        total_sales = df['monetary'].sum()
        avg_sales_per_customer = total_sales / total_customers if total_customers > 0 else 0
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Total Net Sales", f"${total_sales:,.2f}")
        col3.metric("Avg. Sales per Customer", f"${avg_sales_per_customer:,.2f}")

    st.divider()
    
    c1, c2 = st.columns((6, 4))
    with c1:
        st.subheader("🏪 Store Performance")
        store_data = fetch_data("performance/stores")
        if store_data:
            store_df = pd.DataFrame.from_dict(store_data, orient='index').reset_index().rename(columns={'index':'Shopping Mall'})
            fig = px.bar(store_df, x='total_net_sales', y='Shopping Mall', orientation='h', title='Total Net Sales by Store')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("💳 Payment Method Distribution")
        payment_data = fetch_data("insights/payment-methods")
        if payment_data:
            payment_df = pd.DataFrame(payment_data)
            fig = px.pie(payment_df, values='proportion', names='payment_method', title='Payment Method Usage', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Customer Segmentation ---
with tab2:
    st.header("👥 K-Means Customer Segmentation")
    st.markdown("Customers are segmented into 5 distinct groups based on their Recency, Frequency, and Monetary (RFM) behavior using K-Means clustering.")
    
    if cluster_data:
        cluster_df = pd.DataFrame(cluster_data)
        
        c1, c2 = st.columns((7, 3))
        with c1:
            st.subheader("3D Cluster Visualization")
            fig_3d = px.scatter_3d(
                cluster_df,
                x='recency', y='frequency', z='monetary',
                color='cluster',
                hover_data=['customer_id'],
                title='RFM Clusters in 3D Space'
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        with c2:
            st.subheader("Segment Distribution")
            segment_counts = cluster_df['cluster'].value_counts().reset_index()
            fig_bar = px.bar(segment_counts, x='count', y='cluster', orientation='h', title="Customers per Cluster")
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Cluster Profiles")
        st.markdown("Average RFM values for each cluster. This helps define the characteristics of each segment.")
        profiles_data = fetch_data("segmentation/cluster-profiles")
        if profiles_data:
            profiles_df = pd.DataFrame(profiles_data)
            st.dataframe(profiles_df.style.background_gradient(cmap='viridis'), use_container_width=True)

# --- Tab 3: Category & Trends ---
with tab3:
    st.header("📅 Sales Trends & 🛍️ Category Insights")
    
    st.subheader("Monthly Sales Trend")
    monthly_sales_data = fetch_data("trends/monthly-sales")
    if monthly_sales_data:
        monthly_df = pd.DataFrame(monthly_sales_data)
        monthly_df['invoice_date'] = pd.to_datetime(monthly_df['invoice_date'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_df['invoice_date'], y=monthly_df['net_sales'], mode='lines+markers', name='Sales'))
        fig.update_layout(title='Monthly Net Sales Over Time', xaxis_title='Date', yaxis_title='Net Sales ($)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Category Heatmap by Customer Cluster")
    category_data = fetch_data("insights/category-by-cluster")
    if category_data:
        cat_df = pd.DataFrame(category_data)
        heatmap_df = cat_df.pivot_table(index='category', columns='cluster', values='net_sales', fill_value=0)
        fig_heatmap = px.imshow(heatmap_df, text_auto=True, aspect="auto", title="Total Sales per Category Across Clusters")
        st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Tab 4: Campaign Simulator ---
with tab4:
    st.header("🎯 Campaign ROI Simulator")
    st.markdown("Select a customer cluster and a discount percentage to project the potential ROI of a targeted marketing campaign.")
    
    if cluster_data:
        cluster_list = sorted(pd.DataFrame(cluster_data)['cluster'].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            target_cluster = st.selectbox("Select Target Cluster", options=cluster_list)
        with col2:
            discount_pct = st.slider("Select Campaign Discount (%)", 1, 40, 10)
        
        if st.button("🚀 Run Simulation"):
            sim_params = {"target_cluster": target_cluster, "discount": discount_pct / 100.0}
            with st.spinner("Running simulation..."):
                sim_results = fetch_data("simulations/campaign", params=sim_params)
            
            if sim_results:
                st.subheader("Projected Campaign Results")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Targeted Customers", f"{sim_results['customer_count']:,}")
                kpi2.metric("Projected Campaign Cost", sim_results['campaign_cost'])
                kpi3.metric("Projected ROI", sim_results['projected_roi_percent'])
                
                st.success(f"A campaign targeting Cluster {target_cluster} with a {discount_pct}% discount is projected to yield an ROI of **{sim_results['projected_roi_percent']}**.")

