from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
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
num_clusters = 4  # Adjust this based on your analysis

# Run K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
rfm_normalized['Cluster'] = kmeans.fit_predict(rfm_normalized)

# Add cluster labels to the original dataset
rfm_filtered['Cluster'] = rfm_normalized['Cluster']

sns.pairplot(rfm_normalized, hue='Cluster', palette='tab10')
plt.suptitle("RFM Clusters")
plt.show()

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