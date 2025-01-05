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

# Layout with vertical navigation and grid content
app.layout = html.Div([
    # Vertical navigation panel
    html.Div(
        className="nav-panel",
        children=[
            html.H2("Dashboard", style={'marginBottom': '30px'}),
            html.A("Time Series Analysis", href="#time-series", className="nav-link"),
            html.A("Sales by Country", href="#sales-country", className="nav-link"),
            html.A("Top Products", href="#top-products", className="nav-link"),
            html.A("Customer Clustering", href="#clustering", className="nav-link"),
        ]
    ),

    # Main content area with grid layout
    html.Div(
        className="content",
        children=[
            # Raw Time Series Card
            html.Div(
                className="card card-large",
                id="time-series-raw",
                children=[
                    html.H3("Raw Time Series"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.line(
                                x=time_series.index,
                                y=time_series.values,
                                title="Raw Time Series",
                                template="plotly_dark"
                            ).update_layout(
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    ),
                ]
            ),

            # Trend Component Card
            html.Div(
                className="card",
                id="time-series-trend",
                children=[
                    html.H3("Trend Component"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.line(
                                x=trend.index,
                                y=trend.values,
                                title="Trend Component",
                                template="plotly_dark"
                            ).update_layout(
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    ),
                ]
            ),

            # Sales by Country Card
            html.Div(
                className="card",
                id="sales-country",
                children=[
                    html.H3("Sales by Country"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.choropleth(
                                sales_country, locations="Country",
                                locationmode="country names", color="Revenue",
                                title="Sales by Country", template="plotly_dark"
                            )),
                        ]
                    ),
                ]
            ),

            # Top Products Card
            html.Div(
                className="card",
                id="top-products",
                children=[
                    html.H3("Top Products by Revenue"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.bar(
                                top_products_data, x='Description', y='Revenue',
                                title="Top Products", template="plotly_dark"
                            )),
                        ]
                    ),
                ]
            ),

            # Customer Clustering Card (Larger)
            html.Div(
                className="card card-large",
                id="clustering",
                children=[
                    html.H3("Customer Clustering"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.scatter_3d(
                                rfm_clustered, x='Recency', y='Frequency', z='Monetary',
                                color='Cluster', title="3D Clustering Visualization", template="plotly_dark"
                            )),
                        ]
                    ),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.bar(
                                cluster_summary, x=cluster_summary.index, y='Cluster_size',
                                title="Cluster Sizes", template="plotly_dark"
                            )),
                        ]
                    ),
                ]
            ),
        ]
    ),
])


if __name__ == '__main__':
    app.run_server(debug=True)
