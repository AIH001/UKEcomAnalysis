# main.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from ucimlrepo import fetch_ucirepo
def load_and_prepare_data():
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


def cluster_rfm(rfm_normalized, rfm_dataset, num_clusters=4):

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    rfm_normalized['Cluster'] = kmeans.fit_predict(rfm_normalized[['Recency', 'Frequency', 'Monetary']])

    # Ensure rfm_dataset has CustomerID as a column, not as an index
    rfm_dataset = rfm_dataset.reset_index()

    # Merge clusters back into the original RFM dataset
    rfm_combined = pd.merge(rfm_dataset, rfm_normalized[['Cluster']], left_index=True, right_index=True)

    return rfm_combined, rfm_normalized


def cluster_breakdown(rfm_combined):

    # Breakdown metrics by Cluster
    cluster_summary = rfm_combined.groupby('Cluster').agg(
        Recency_avg=('Recency', 'mean'),
        Frequency_avg=('Frequency', 'mean'),
        Monetary_avg=('Monetary', 'mean'),
        Recency_median=('Recency', 'median'),
        Frequency_median=('Frequency', 'median'),
        Monetary_median=('Monetary', 'median'),
        Cluster_size=('CustomerID', 'count')  # Count customers in each cluster
    ).reset_index()
    return cluster_summary

def cluster_statistics_visualization(cluster_summary):
    cluster_summary.reset_index(inplace=True)
    # Convert cluster_summary.index to string to enforce discrete behavior
    cluster_summary_reset = cluster_summary.reset_index()  # Reset index for easy access
    cluster_summary_reset['index'] = cluster_summary_reset['index'].astype(str)  # Convert index to string

    # Bar chart for Recency
    recency_fig = px.bar(
        cluster_summary_reset,
        x='index',
        y='Recency_avg',
        color='index',  # Use the index column for coloring
        color_discrete_sequence=[
            '#A7C7E7',
            '#6082B6',
            '#87CEEB',
            '#4682B4'
        ],
        title="Average Recency by Cluster",
        labels={"index": "Cluster", "Recency_avg": "Average Recency"}
    )

    # Bar chart for Frequency
    frequency_fig = px.bar(
        cluster_summary_reset,
        x='index',
        y='Frequency_avg',
        color='index',
        color_discrete_sequence=[
            '#A7C7E7',
            '#6082B6',
            '#87CEEB',
            '#4682B4'
        ],
        title="Average Frequency by Cluster",
        labels={"index": "Cluster", "Frequency_avg": "Average Frequency"}
    )

    # Bar chart for Monetary
    monetary_fig = px.bar(
        cluster_summary_reset,
        x='index',
        y='Monetary_avg',
        color='index',
        color_discrete_sequence=[
            '#A7C7E7',
            '#6082B6',
            '#87CEEB',
            '#4682B4'
        ],
        title="Average Monetary Value by Cluster",
        labels={"index": "Cluster", "Monetary_avg": "Average Monetary Value"}
    )

    return recency_fig, frequency_fig, monetary_fig,


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


def calculate_return_rate(df):

    df['Returned'] = df['Quantity'] < 0  # Assume negative quantity means return
    return_rate = df['Returned'].value_counts(normalize=True) * 100
    labels = ['Not Returned', 'Returned']

    fig = go.Figure(data=[go.Pie(labels=labels, values=return_rate, hole=0.5)])
    fig.update_traces(
        marker=dict(
            colors=[
                '#6082B6',  # Custom dark background for slice 1
                'red',  # Default red color for slice 3 (or wherever required)
            ]
        )
    )
    return fig


def most_returned_products(df, top_n=5):
    # Filter for returned products
    returned_products = df[df['Quantity'] < 0]

    # Aggregate and sort by return quantity
    top_returned = (
        returned_products.groupby('Description')['Quantity']
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Truncate descriptions for x-axis labels
    truncated_labels = [desc[:15] + "..." if len(desc) > 15 else desc for desc in top_returned.index]

    # Create the bar chart
    fig = px.bar(
        top_returned,
        x=truncated_labels,
        y=top_returned.values,
        labels={"x": "Product", "y": "Return Quantity"}
    )

    # Add hover data with full descriptions
    fig.update_traces(
        hovertemplate='<b>Product:</b> %{customdata[0]}<br><b>Return Quantity:</b> %{y}',
        customdata=[top_returned.index],  # Full descriptions as hover data
        marker_color="#6082B6"
    )

    # Customize layout
    fig.update_layout(
        xaxis_title="Product",
        yaxis_title="Return Quantity",
        xaxis=dict(tickangle=-45),  # Rotate x-axis labels if needed
    )

    return fig


def average_sales_by_time(df):

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Sales'] = df['Quantity'] * df['UnitPrice']

    # Total Sales for each period
    total_sales = df['Sales'].sum()

    # Total unique days, weeks, and months in the data
    total_days = len(df['InvoiceDate'].dt.date.unique())
    total_weeks = len(df['InvoiceDate'].dt.to_period('W').unique())
    total_months = len(df['InvoiceDate'].dt.to_period('M').unique())

    # Calculate averages
    avg_day = total_sales / total_days
    avg_week = total_sales / total_weeks
    avg_month = total_sales / total_months

    return avg_day, avg_week, avg_month

def top_performing_hours(df):
    # Ensure InvoiceDate is in datetime format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Extract the hour and calculate sales
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['Sales'] = df['Quantity'] * df['UnitPrice']

    # Group by hour and aggregate sales
    sales_by_hour = df.groupby('Hour')['Sales'].sum()

    # Create a mapping for hour labels
    hour_labels = {0: '12 AM', 1: '1 AM', 2: '2 AM', 3: '3 AM', 4: '4 AM', 5: '5 AM',
                   6: '6 AM', 7: '7 AM', 8: '8 AM', 9: '9 AM', 10: '10 AM', 11: '11 AM',
                   12: '12 PM', 13: '1 PM', 14: '2 PM', 15: '3 PM', 16: '4 PM', 17: '5 PM',
                   18: '6 PM', 19: '7 PM', 20: '8 PM', 21: '9 PM', 22: '10 PM', 23: '11 PM'}

    # Replace numeric hours with meaningful labels
    sales_by_hour.index = sales_by_hour.index.map(hour_labels)

    # Create the bar chart
    fig = px.bar(sales_by_hour, x=sales_by_hour.index, y=sales_by_hour.values)
    fig.update_traces(marker_color="#6082B6")
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Total Sales")

    return fig
