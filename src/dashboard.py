import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Retail & Customer Insights Dashboard",
    page_icon="🛒"
)

# # --- API Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"


# To this placeholder (we will get the real URL after deploying the API):
#API_BASE_URL = "https://retail-api-gajalakshmi.onrender.com/" 


# --- Helper Functions ---
@st.cache_data
def fetch_data(endpoint: str, params: dict = None):
    """Fetches data from the FastAPI endpoint with caching."""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API: {e}. Please ensure the API server is running.")
        return None

# --- Dashboard Pages ---
def overview_page():
    st.title("📈 Dashboard Overview")
    st.markdown("Key performance indicators for the retail business.")
    
    col1, col2, col3 = st.columns(3)
    rfm_data = fetch_data("customers/rfm-segments")
    if rfm_data and isinstance(rfm_data, list):
        total_customers = len(rfm_data)
        total_sales = pd.DataFrame(rfm_data)['monetary'].sum()
        avg_sales = total_sales / total_customers if total_customers > 0 else 0
        
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Total Net Sales", f"${total_sales:,.2f}")
        col3.metric("Avg. Sales per Customer", f"${avg_sales:,.2f}")

    st.header("🏪 Store Performance at a Glance")
    store_data = fetch_data("performance/stores")
    if store_data:
        store_df = pd.DataFrame.from_dict(store_data, orient='index').reset_index()
        store_df.columns = ['Shopping Mall', 'Total Net Sales', 'Average Net Sales']
        fig = px.bar(store_df, x='Shopping Mall', y='Total Net Sales', title='Total Sales by Store')
        st.plotly_chart(fig, use_container_width=True)


def customer_deep_dive_page():
    st.title("👥 Customer Deep Dive")
    
    st.header("RFM Segmentation")
    rfm_data = fetch_data("customers/rfm-segments")
    if rfm_data and isinstance(rfm_data, list):
        rfm_df = pd.DataFrame(rfm_data)
        segment_counts = rfm_df['segment_name'].value_counts()
        fig_pie = px.pie(segment_counts, values=segment_counts.values, names=segment_counts.index, title='Customer Segment Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)
        st.dataframe(rfm_df)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.header("High-Value vs. Low-Value Customers")
        value_data = fetch_data("customers/value-segmentation")
        if value_data:
            value_df = pd.DataFrame(value_data)
            fig = px.pie(value_df, values='count', names='value_segment', title='High-Value vs. Low-Value Segments')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.header("Repeat vs. One-Time Customers")
        repeat_data = fetch_data("customers/repeat-vs-onetime")
        if repeat_data:
            repeat_df = pd.DataFrame(repeat_data)
            fig = px.pie(repeat_df, values='net_sales', names='customer_type', title='Sales Contribution: Repeat vs. One-Time')
            st.plotly_chart(fig, use_container_width=True)

    st.header("🏆 Top 10% Customers by Purchase Value")
    top_customers_data = fetch_data("customers/top-customers")
    if top_customers_data:
        st.dataframe(pd.DataFrame(top_customers_data))


def profitability_and_trends_page():
    st.title("💰 Profitability and Sales Trends")

    st.header("Discount Impact on Profitability by Category")
    discount_data = fetch_data("insights/discount-impact")
    if discount_data:
        discount_df = pd.DataFrame(discount_data)
        fig = px.bar(discount_df, y='category', x=['net_sales', 'total_discount'], 
                     title='Net Sales vs. Discounts by Category', orientation='h', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    st.header("📅 Monthly and Quarterly Sales Trends")
    seasonality_data = fetch_data("insights/seasonality")
    if seasonality_data:
        season_df = pd.DataFrame(seasonality_data)
        season_df['invoice_date'] = pd.to_datetime(season_df['invoice_date'])
        season_df['quarter'] = season_df['invoice_date'].dt.to_period('Q').astype(str)
        quarterly_sales = season_df.groupby('quarter')['net_sales'].sum().reset_index()
        
        fig_monthly = px.line(season_df, x='invoice_date', y='net_sales', title='Monthly Sales Over Time', markers=True)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        fig_quarterly = px.bar(quarterly_sales, x='quarter', y='net_sales', title='Quarterly Sales')
        st.plotly_chart(fig_quarterly, use_container_width=True)


def category_insights_page():
    st.title("🛍️ Category & Payment Insights")
    st.header("Category Performance by Customer Segment")
    category_data = fetch_data("insights/category-by-segment")
    if category_data:
        category_df = pd.DataFrame(category_data)
        fig = px.sunburst(category_df, path=['segment_name', 'category'], values='net_sales',
                          title='Category Spending Across Top Customer Segments')
        st.plotly_chart(fig, use_container_width=True)
    
    st.header("💳 Payment Method Preferences")
    payment_data = fetch_data("insights/payment-methods")
    if payment_data:
        payment_df = pd.DataFrame(payment_data)
        fig = px.pie(payment_df, values='proportion', names='payment_method', title='Payment Method Distribution')
        st.plotly_chart(fig, use_container_width=True)


def campaign_simulation_page():
    st.title("🎯 Campaign ROI Simulation")
    st.markdown("Model the potential return on investment for a targeted marketing campaign.")
    
    rfm_data = fetch_data("customers/rfm-segments")
    if rfm_data and isinstance(rfm_data, list):
        segments = pd.DataFrame(rfm_data)['segment_name'].unique()
        
        col1, col2 = st.columns(2)
        with col1:
            target_segment = st.selectbox("Select Target Segment", segments, index=list(segments).index('Champions'))
        with col2:
            discount = st.slider("Select Campaign Discount (%)", 1, 50, 10)
            
        if st.button("Run Simulation"):
            sim_params = {"target_segment": target_segment, "discount": discount / 100.0}
            sim_results = fetch_data("simulations/campaign", params=sim_params)
            
            if sim_results:
                st.subheader("Projected Campaign Results")
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Targeted Customers", f"{sim_results['customer_count']:,}")
                kpi2.metric("Projected Campaign Cost", f"${sim_results['campaign_cost']:,.2f}")
                kpi3.metric("Projected ROI", f"{sim_results['projected_roi']:,.2f}%")
                st.success(f"This campaign targets {sim_results['customer_count']} customers in the '{target_segment}' segment. With a {discount}% discount, the projected ROI is {sim_results['projected_roi']:.2f}%.")


# --- Main App Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Customer Deep Dive",
    "Profitability & Trends",
    "Category & Payment Insights",
    "Campaign Simulation"
])

if page == "Overview":
    overview_page()
elif page == "Customer Deep Dive":
    customer_deep_dive_page()
elif page == "Profitability & Trends":
    profitability_and_trends_page()
elif page == "Category & Payment Insights":
    category_insights_page()
elif page == "Campaign Simulation":
    campaign_simulation_page()
