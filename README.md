# MLOps Capstone Project: Retail Store & Customer Insights

This project is an end-to-end analytics and machine learning pipeline designed to provide actionable insights into a retail chain's store performance and customer behavior. It processes daily sales data, segments customers using an ML model, and serves the findings through a REST API and an interactive web dashboard.

---

## 📋 Table of Contents
* [Project Overview](#-project-overview)
* [Key Features](#-key-features)
* [Tech Stack](#-tech-stack)
* [Project Structure](#-project-structure)
* [Setup and Installation](#-setup-and-installation)
* [How to Run the Project](#-how-to-run-the-project)
* [API Endpoints](#-api-endpoints)
* [Dashboard Pages](#-dashboard-pages)

---

## 🎯 Project Overview

The primary goal of this project is to build a robust MLOps pipeline that transforms raw retail data into strategic business insights. The system automates data processing, performs RFM-based customer segmentation with K-Means clustering, and analyzes key business metrics like store performance, profitability, and seasonal trends. These insights are then made accessible to stakeholders via a FastAPI backend and a Streamlit-powered dashboard.



---

## ✨ Key Features

* **Automated Data Pipeline:** Cleans, transforms, and enriches raw sales data automatically.
* **ML-Powered Customer Segmentation:** Uses **RFM (Recency, Frequency, Monetary)** analysis combined with a **K-Means clustering** model to identify distinct customer segments like "Champions," "Loyal," and "At Risk."
* **Performance Analytics:** Compares sales volume, revenue, and transaction counts across different stores.
* **Profitability Insights:** Calculates the impact of discounts on net sales for each product category.
* **Trend Analysis:** Identifies monthly and quarterly sales patterns to understand seasonality.
* **Interactive Dashboard:** A user-friendly web application built with Streamlit for visualizing all key metrics and insights.
* **REST API:** A scalable backend built with FastAPI to serve the processed data and analytics results.
* **Campaign Simulation:** An interactive tool to model the potential ROI of marketing campaigns targeted at specific customer segments.

---

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Uvicorn, Gunicorn
* **Frontend (Dashboard):** Streamlit
* **Data Manipulation & ML:** Pandas, Scikit-learn
* **Visualization:** Plotly
* **Deployment:** `Procfile` for cloud platforms like Render or Heroku

---

## 📁 Project Structure

```
/MLOps_Capstone_Project/
|-- data/
|   |-- customer_shopping_data.csv      # Raw input data
|   |-- processed_customer_data.parquet # Processed output data
|-- src/
|   |-- __init__.py
|   |-- data_processing.py              # Script for data ingestion and cleaning
|   |-- analysis.py                     # Core analytics and ML modeling functions
|   |-- api.py                          # FastAPI application
|   |-- dashboard.py                    # Streamlit dashboard application
|-- Procfile                            # Deployment configuration
|-- requirements.txt                    # Python dependencies
|-- README.md                           # This file
```

---
### Process to Run it on Local VM:
## ⚙️ Setup and Installation

Follow these steps to set up the project environment on your local machine.

### Prerequisites
* Python 3.8 or higher
* pip package manager

### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd MLOps_Capstone_Project
    ```

2.  **Create and activate a virtual environment:**
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Run the Project

The project consists of three main components that need to be run in order.

### Step 1: Process the Data
First, run the data processing script to generate the `processed_customer_data.parquet` file from the raw CSV.

```bash
python src/data_processing.py
```

### Step 2: Start the FastAPI Server
Next, start the API server, which will serve the data to the dashboard.

```bash
uvicorn src.api:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can access the auto-generated documentation at `http://127.0.0.1:8000/docs`.

### Step 3: Launch the Streamlit Dashboard
Finally, in a **new terminal**, run the Streamlit dashboard.

```bash
streamlit run src/dashboard.py
```
The interactive dashboard will open in your web browser, usually at `http://localhost:8501`.

---

## 🌐 API Endpoints

The FastAPI server exposes the following endpoints:

| Method | Endpoint                          | Description                                         |
| :----- | :-------------------------------- | :-------------------------------------------------- |
| `GET`  | `/performance/stores`             | Get sales performance data for all stores.          |
| `GET`  | `/customers/rfm-segments`         | Get RFM and K-Means segment data for all customers. |
| `GET`  | `/insights/seasonality`           | Get monthly and quarterly sales trend data.         |
| `GET`  | `/insights/discount-impact`       | Get profitability and discount data by category.    |
| `GET`  | `/customers/repeat-vs-onetime`    | Compare sales from repeat vs. one-time customers.   |
| `GET`  | `/simulations/campaign`           | Run a campaign ROI simulation for a target segment. |

---

## 📊 Dashboard Pages

The Streamlit dashboard is organized into several pages for easy navigation:

* **Overview:** A high-level summary of key metrics like total sales, total customers, and store performance.
* **Customer Analysis:** Deep dive into customer segments created by the K-Means model, with visualizations of their characteristics and value.
* **Performance & Trends:** Visual analysis of seasonal sales patterns and the impact of discounts on profitability.
* **Campaign Simulator:** An interactive tool for business strategists to model the financial outcome of targeted marketing campaigns.

### Links to access the API & Dashboard App:
**Retail_Insights_API** - [https://retail-insights-api-gajalakshmi-j.onrender.com/docs]

**Retail_Sales_Insights_Dashboard** - [https://retail-sales-insights-dashboard-gajalakshmi-j.streamlit.app]
