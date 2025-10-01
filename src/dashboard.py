# src/dashboard.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Retail Insights Dashboard",
    page_icon="🛒"
)

# --- API Configuration ---
API_BASE_URL = "https://retail-insights-api-gajalakshmi-j.onrender.com"

# --- Helper Function to Fetch Data ---
@st.cache_data(ttl=300)
def fetch_data(endpoint: str, params: dict = None):
    """Fetches data from the FastAPI endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {e}. Please ensure the API server is running.")
        return None

# --- Dashboard Header ---
st.title("✅ Retail Analytics Dashboard")
st.markdown("A smart retail analytics dashboard that empowers teams to make confident decisions based on customer shopping behavior.")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", [
    "Overview",
    "Customer Analysis",
    "Performance & Trends",
    "Campaign Simulator"
])

# --- Pre-fetch essential data ---
rfm_data = fetch_data("customers/rfm-segments")

# --- Page 1: Overview ---
if page == "Overview":
    st.header("📈 Business Overview")
    st.markdown("A high-level summary of key business metrics.")

    if rfm_data:
        df = pd.DataFrame(rfm_data)
        total_customers = df['customer_id'].nunique()
        total_sales = df['monetary'].sum()
        avg_sales_per_customer = total_sales / total_customers

        # --- Key Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Total Net Sales", f"${total_sales:,.2f}")
        col3.metric("Avg. Sales per Customer", f"${avg_sales_per_customer:,.2f}")

    st.divider()

    # --- Store & Payment Analysis ---
    st.subheader("Store & Payment Insights")
    col1, col2 = st.columns(2)
    with col1:
        store_data = fetch_data("performance/stores")
        if store_data:
            store_df = pd.DataFrame.from_dict(store_data, orient='index').reset_index().rename(columns={'index': 'Store', 'sum': 'Total Sales'})
            fig = px.bar(store_df, y='Store', x='Total Sales', orientation='h', title='Total Sales by Store')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** The horizontal bar chart makes it easy to quickly identify top and bottom-performing stores for strategic focus.")

    with col2:
        payment_data = fetch_data("insights/payment-methods")
        if payment_data:
            payment_df = pd.DataFrame(payment_data)
            fig = px.pie(payment_df, values='proportion', names='payment_method', title='Payment Method Distribution', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** Understanding the primary payment methods helps in optimizing the checkout process and planning partnerships (e.g., credit card offers).")

# --- Page 2: Customer Analysis ---
elif page == "Customer Analysis":
    st.header("👥 Customer Segmentation (RFM & K-Means)")
    st.markdown("Customers are grouped into segments using a **K-Means machine learning model** based on their shopping behavior.")
    
    if rfm_data:
        rfm_df = pd.DataFrame(rfm_data)
        
        # --- Simplified K-Means Visualization ---
        st.subheader("Customer Segment Distribution")
        segment_counts = rfm_df['segment'].value_counts().reset_index()
        fig_dist = px.bar(segment_counts, x='segment', y='count', title="Number of Customers in Each Segment")
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("**Insight:** This shows the size of each customer segment. A large 'Champions' segment is healthy, while a large 'At Risk' segment requires immediate attention.")

        st.subheader("Segment Profiles (Characteristics)")
        segment_profiles = rfm_df.groupby('segment')[['recency', 'frequency', 'monetary']].mean()
        
        # Create 3 separate plots to solve the scaling issue
        fig = make_subplots(rows=1, cols=3, subplot_titles=('Avg. Recency (Days)', 'Avg. Frequency (Visits)', 'Avg. Monetary (Spend)'))
        
        fig.add_trace(go.Bar(x=segment_profiles.index, y=segment_profiles['recency'], name='Recency'), row=1, col=1)
        fig.add_trace(go.Bar(x=segment_profiles.index, y=segment_profiles['frequency'], name='Frequency'), row=1, col=2)
        fig.add_trace(go.Bar(x=segment_profiles.index, y=segment_profiles['monetary'], name='Monetary'), row=1, col=3)
        
        fig.update_layout(height=400, title_text="Comparing RFM Values Across Segments", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""**Insight:** The most significant finding from this analysis is the lack of repeat business, as the average frequency for every segment is just one. The primary strategic focus for the company should be on customer retention and implementing tactics to encourage a second purchase, especially from the "Loyal" and "Champions" segments.""")

        st.divider()
        st.subheader("Repeat vs. One-Time Customers")
        repeat_data = fetch_data("customers/repeat-vs-onetime")
        if repeat_data:
            repeat_df = pd.DataFrame(repeat_data)
            fig = px.pie(repeat_df, values='total_sales', names='customer_type', title="Sales Contribution: Repeat vs. One-Time", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Insight:** The single most critical insight is that the business is failing at customer retention, as the data shows that virtually no customers—regardless of how much they spend—ever make a second purchase..")

# --- Page 3: Performance & Trends ---
elif page == "Performance & Trends":
    st.header("💰 Performance & Seasonal Trends")
    st.markdown("Analyzing profitability and sales patterns over time.")

    st.subheader("Profitability by Product Category")
    discount_data = fetch_data("insights/discount-impact")
    if discount_data:
        discount_df = pd.DataFrame(discount_data)
        fig = go.Figure(data=[
            go.Bar(name='Net Sales', x=discount_df['category'], y=discount_df['net_sales']),
            go.Bar(name='Discount Given', x=discount_df['category'], y=discount_df['total_discount'])
        ])
        fig.update_layout(barmode='group', title="Net Sales vs. Discounts by Category")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** This grouped bar chart clearly shows which categories rely heavily on discounts versus those that are profitable with minimal discounting.")
    
    st.divider()
    
    st.subheader("Monthly Sales Trends")
    season_data = fetch_data("insights/seasonality")
    if season_data:
        monthly_df = pd.DataFrame(season_data['monthly'])
        monthly_df['invoice_date'] = pd.to_datetime(monthly_df['invoice_date'])
        fig_monthly = px.line(monthly_df, x='invoice_date', y='net_sales', title='Monthly Sales Over Time', markers=True)
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.markdown("**Insight:** The line chart reveals seasonal peaks and troughs, which is critical for inventory management and planning marketing campaigns.")
        
# --- Page 4: Campaign Simulator ---
elif page == "Campaign Simulator":
    st.header("🎯 Campaign ROI Simulator")
    st.markdown("Model the potential return on investment for a targeted marketing campaign.")
    
    if rfm_data:
        segments = sorted(pd.DataFrame(rfm_data)['segment'].unique())
        
        st.info("Select a customer segment and a discount percentage to project the campaign's financial outcome.", icon="ℹ️")
        
        col1, col2 = st.columns(2)
        with col1:
            target_segment = st.selectbox("Select Target Segment", segments, index=segments.index("Champions"))
        with col2:
            discount = st.slider("Select Campaign Discount (%)", 5, 40, 10)
            
        if st.button("🚀 Run Simulation"):
            sim_params = {"target_segment": target_segment, "discount_pct": discount / 100.0}
            sim_results = fetch_data("simulations/campaign", params=sim_params)
            
            if sim_results:
                st.subheader("Projected Campaign Results")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Targeted Customers", f"{sim_results['targeted_customers']:,}")
                kpi2.metric("Projected Campaign Cost", f"${sim_results['campaign_cost']:,.2f}")
                kpi3.metric("Projected ROI", f"{sim_results['projected_roi_percent']:,.2f}%")
                

                st.success(f"**Conclusion:** Targeting the '{target_segment}' segment with a {discount}% discount is projected to yield a significant return on investment.")
