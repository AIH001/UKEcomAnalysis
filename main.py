# eda.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

def load_and_prepare_data():
    from ucimlrepo import fetch_ucirepo
    online_retail = fetch_ucirepo(id=352)
    ids = online_retail.data['ids']
    X = online_retail.data.features
    X = pd.concat([X, ids[['InvoiceNo']]], axis=1)

    # Process data
    df = X.copy()
    df['CustomerID'] = df['CustomerID'].replace("", float('nan'))
    df = df[df['CustomerID'].notnull()]
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def calculate_rfm(df):
    rfm_dataset = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'Revenue': 'sum'  # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Revenue': 'Monetary'})

    scaler = MinMaxScaler()
    rfm_normalized = scaler.fit_transform(rfm_dataset)
    rfm_normalized = pd.DataFrame(rfm_normalized, columns=['Recency', 'Frequency', 'Monetary'])
    return rfm_dataset, rfm_normalized

def cluster_rfm(rfm_normalized, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    rfm_normalized['Cluster'] = kmeans.fit_predict(rfm_normalized)
    return rfm_normalized

def cluster_breakdown(rfm_dataset, rfm_clustered):
    # Combine clusters with original RFM metrics
    rfm_combined = rfm_dataset.copy()
    rfm_combined['Cluster'] = rfm_clustered['Cluster']

    # Breakdown metrics
    cluster_summary = rfm_combined.groupby('Cluster').agg(
        Recency_avg=('Recency', 'mean'),
        Frequency_avg=('Frequency', 'mean'),
        Monetary_avg=('Monetary', 'mean'),
        Recency_median=('Recency', 'median'),
        Frequency_median=('Frequency', 'median'),
        Monetary_median=('Monetary', 'median'),
        Cluster_size=('Cluster', 'count')
    )
    return cluster_summary

def time_series_analysis(df):
    time_series = df.set_index('InvoiceDate')['Revenue'].resample('D').sum()

    # Decompose the time series
    decomposition = seasonal_decompose(time_series, model='additive', period=30)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return time_series, trend, seasonal, residual

def sales_by_country(df):
    sales_country = df.groupby('Country')['Revenue'].sum().reset_index()
    return sales_country

def top_products(df, n=10):
    top_products = df.groupby('Description')['Revenue'].sum().reset_index()
    top_products = top_products.sort_values(by='Revenue', ascending=False).head(n)
    return top_products
