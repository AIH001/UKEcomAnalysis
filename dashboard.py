import dash
from dash import dcc, html
from main import load_and_prepare_data, calculate_rfm, cluster_rfm, cluster_breakdown, time_series_analysis, sales_by_country, top_products
import plotly.express as px

# Load and analyze data
df = load_and_prepare_data()
rfm_dataset, rfm_normalized = calculate_rfm(df)
rfm_clustered = cluster_rfm(rfm_normalized)
cluster_summary = cluster_breakdown(rfm_dataset, rfm_clustered)
time_series, trend, seasonal, residual = time_series_analysis(df)
sales_country = sales_by_country(df)
top_products_data = top_products(df)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Online Retail Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs([
        # Tab for Time Series Analysis
        dcc.Tab(label='Time Series Analysis', className='tab', selected_className='tab-selected', children=[
            html.H2("Revenue Over Time"),
            dcc.Graph(figure=px.line(x=time_series.index, y=time_series.values, title="Raw Time Series", template="plotly_dark")),
            dcc.Graph(figure=px.line(x=trend.index, y=trend.values, title="Trend Component", template="plotly_dark")),
        ]),

        # Tab for Sales by Country
        dcc.Tab(label='Sales by Country', className='tab', selected_className='tab-selected', children=[
            html.H2("Sales by Country"),
            dcc.Graph(figure=px.choropleth(sales_country, locations="Country", locationmode="country names",
                                           color="Revenue", title="Sales by Country", template="plotly_dark")),
        ]),

        # Tab for Top Products
        dcc.Tab(label='Top Products', className='tab', selected_className='tab-selected', children=[
            html.H2("Top Products by Revenue"),
            dcc.Graph(figure=px.bar(top_products_data, x='Description', y='Revenue', title="Top Products", template="plotly_dark")),
        ]),

        # Tab for Clustering
        dcc.Tab(label='Customer Clustering', className='tab', selected_className='tab-selected', children=[
            html.H2("Customer Segmentation"),
            dcc.Graph(figure=px.scatter_3d(rfm_clustered, x='Recency', y='Frequency', z='Monetary', color='Cluster',
                                           title="3D Clustering Visualization", template="plotly_dark")),
            html.H3("Cluster Breakdown"),
            dcc.Graph(figure=px.bar(cluster_summary, x=cluster_summary.index, y='Cluster_size', title="Cluster Sizes",
                                    template="plotly_dark")),
        ]),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
