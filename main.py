from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from scipy.fft import fft
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# fetch dataset
online_retail = fetch_ucirepo(id=352)
ids = online_retail.data['ids']
data_type = {'CustomerID': str}

# variable information
#print(online_retail.variables)

# data (as pandas dataframes)
X = online_retail.data.features
X = pd.concat([X, ids[['InvoiceNo']]], axis=1)  # Add InvoiceNo as a new column into a new df
# no Y (no targets)

# df processing, removing null values and calculating revenue
df = X.copy()
df['CustomerID'] = df['CustomerID'].replace("", float('nan'))
df = df[df['CustomerID'].notnull()]
df['Revenue'] = df['Quantity'] * df['UnitPrice']


# Change CustomerID to string
df = df.astype(data_type)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

columns = ['InvoiceNo', 'InvoiceDate', 'CustomerID', 'Revenue']
df_dataset = df[columns]
# customer segmentation (rfm)
rfm_dataset = df_dataset.groupby('CustomerID').agg({
'InvoiceDate': lambda x: (df_dataset['InvoiceDate'].max() - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'Revenue': 'sum'  # Monetary
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Revenue': 'Monetary'
})

print(rfm_dataset)

rfm_metrics = rfm_dataset[['Recency', 'Frequency', 'Monetary']]

# Initialize the filtered dataset
rfm_filtered = rfm_dataset.copy()

# Apply the percentile filter to each variable
for column in ['Recency', 'Frequency', 'Monetary']:
    lower_bound = rfm_dataset[column].quantile(0.01)
    upper_bound = rfm_dataset[column].quantile(0.99)
    rfm_filtered = rfm_filtered[(rfm_filtered[column] >= lower_bound) & (rfm_filtered[column] <= upper_bound)]

scaler = MinMaxScaler()
rfm_normalized = scaler.fit_transform(rfm_filtered)

rfm_normalized = pd.DataFrame(rfm_normalized, columns=['Recency', 'Frequency', 'Monetary'])
print(rfm_normalized.head())

# Define the number of clusters
num_clusters = 4  # Based on WCSS analysis

# Reset indices for consistency
rfm_filtered.reset_index(drop=True, inplace=True)
rfm_normalized.reset_index(drop=True, inplace=True)

# Run K-Means on normalized data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
rfm_normalized['Cluster'] = kmeans.fit_predict(rfm_normalized)

# Add cluster labels to the original dataset
rfm_filtered['Cluster'] = rfm_normalized['Cluster']

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Assign colors based on cluster labels
colors = rfm_normalized['Cluster']

# Plot the clusters
scatter = ax.scatter(
    rfm_normalized['Recency'],
    rfm_normalized['Frequency'],
    rfm_normalized['Monetary'],
    c=colors, cmap='tab10', s=50
)

# Add axis labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('3D Cluster Visualization')

# Add a legend
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)

plt.show()
print(rfm_normalized['Cluster'])

# Create a DataFrame for 3D plotting
rfm_normalized['Cluster'] = rfm_normalized['Cluster'].astype(str)  # Convert to string for Plotly
fig = px.scatter_3d(
    rfm_normalized,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color='Cluster',
    title="3D Cluster Visualization",
    labels={'Cluster': 'Cluster'},
    opacity=0.8
)

# Show the plot
fig.show()

cluster_summary = rfm_filtered.groupby('Cluster').mean()

# Display the characteristics of each cluster
print("Cluster Characteristics (Mean Values):")
print(cluster_summary)

cluster_sizes = rfm_filtered['Cluster'].value_counts()
print("Number of Data Points in Each Cluster:")
print(cluster_sizes)

cluster_summary_median = rfm_filtered.groupby('Cluster').median()
print("Cluster Characteristics (Median Values):")
print(cluster_summary_median)

# Highlight key metrics for each cluster
key_metrics = ['Recency', 'Frequency', 'Monetary']
cluster_key_metrics = rfm_filtered.groupby('Cluster')[key_metrics].mean()
print("Key Metrics for Each Cluster:")
print(cluster_key_metrics)

plt.figure(figsize=(15, 5))
for i, metric in enumerate(key_metrics, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=rfm_filtered, x='Cluster', y=metric)
    plt.title(f"{metric} Distribution by Cluster")
plt.tight_layout()
plt.show()

# resample data by invoiceData for regular daily intervals
time_series = df.set_index('InvoiceDate')['Revenue'].resample('D').sum()  # 'D' = daily, 'W' = weekly, 'M' = monthly

# plot the time series
plt.figure(figsize=(12, 6))
time_series.plot()
plt.title("Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(time_series, model='additive', period=30)  # Adjust period for monthly or seasonal data

# Analyze the components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Check the variance of the residuals
print("Residual Variance:", residual.var())
print("Trend Variance:", trend.dropna().var())
print("Seasonal Variance:", seasonal.var())

# Plot decomposition
decomposition.plot()
plt.show()

time_series_diff = time_series.diff().dropna()


plt.figure(figsize=(12, 6))
plt.plot(time_series_diff)
plt.title("Differenced Time Series")
plt.xlabel("Date")
plt.ylabel("Differenced Revenue")
plt.show()

# Fit ARIMA model
model = ARIMA(time_series, order=(1, 1, 1))  # (p, d, q)
arima_result = model.fit()

# Summary of the model
print(arima_result.summary())

# Forecast the next 30 days
forecast = arima_result.forecast(steps=30)
print(forecast)

# Plot actual and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(time_series, label="Original")
plt.plot(forecast, label="Forecast", linestyle="--", color="red")
plt.title("ARIMA Forecast")
plt.legend()
plt.show()

# Apply Fourier Transform
fft_values = fft(time_series.values)
fft_freqs = np.fft.fftfreq(len(fft_values))

# Plot frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(fft_freqs[1:int(len(fft_freqs)/2)], np.abs(fft_values[1:int(len(fft_values)/2)]))
plt.title("Frequency Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(time_series, label="Revenue")
plt.title("Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.show()


# ACF and PACF plots
plot_acf(time_series, lags=50, title="ACF Plot")
plt.show()

plot_pacf(time_series, lags=50, title="PACF Plot")
plt.show()

# Resample data to weekly and monthly frequency
time_series_weekly = time_series.resample('W').sum()  # Weekly aggregation
time_series_monthly = time_series.resample('ME').sum()  # Monthly aggregation

# Plot the two resampled time series
plt.figure(figsize=(14, 7))

# Weekly Data
plt.subplot(2, 1, 1)
plt.plot(time_series_weekly, label='Weekly Aggregation', color='blue')
plt.title("Weekly Aggregated Revenue")
plt.ylabel("Revenue")
plt.legend()

# Monthly Data
plt.subplot(2, 1, 2)
plt.plot(time_series_monthly, label='Monthly Aggregation', color='green')
plt.title("Monthly Aggregated Revenue")
plt.ylabel("Revenue")
plt.legend()

plt.tight_layout()
plt.show()


