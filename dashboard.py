import dash
from dash import dcc, html, Input, Output
from main import *
import plotly.express as px

# Load and analyze data
df = load_and_prepare_data()
rfm_dataset, rfm_normalized = calculate_rfm(df)
rfm_combined, rfm_normalized = cluster_rfm(rfm_normalized, rfm_dataset)
cluster_summary = cluster_breakdown(rfm_combined)
time_series, trend, seasonal, residual = time_series_analysis(df)
sales_country = sales_by_country(df)
top_products_data = top_products(df)
recency_fig, frequency_fig, monetary_fig, size_fig = cluster_statistics_visualization(cluster_summary)
return_rate_chart = calculate_return_rate(df)
returned_products_chart = most_returned_products(df)
avg_day, avg_week, avg_month = average_sales_by_time(df)
hourly_sales_chart = top_performing_hours(df)

# Initialize Dash app
app = dash.Dash(__name__)

# Navigation Panel
nav_panel = html.Div(
    className="nav-panel",
    children=[
        html.H2("Dashboard", style={'marginBottom': '30px'}),
        html.A("General Analytics", href="/general", className="nav-link"),
        html.A("Customer Segmentation", href="/customer-segmentation", className="nav-link"),
        html.A("Time Series Analysis", href="/time-series", className="nav-link"),
    ]
)

# Page Content
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page_content(pathname):
    # Time Series graph
    if pathname == '/time-series':
        return [
            html.Div(
                className="card card-large",
                id="time-series-raw",
                children=[
                    html.H3("Raw Time Series"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(
                                figure=px.line(
                                    x=time_series.index,
                                    y=time_series.values,
                                    title="Raw Time Series",
                                    template="plotly_dark"
                                ).update_layout(
                                    paper_bgcolor="#1e1e2f",
                                    plot_bgcolor="#1e1e2f",
                                    margin=dict(r=20, t=40, l=20, b=20)
                                )
                            ),
                        ]
                    )

                ]
            ),
            # trend graph
            html.Div(
                className="card card-large",
                id="time-series-trend",
                children=[
                    html.H3("Time Series Trend"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(
                                figure=px.line(
                                    x=trend.index,
                                    y=trend.values,
                                    title="Trend Component",
                                    template="plotly_dark"
                                ).update_layout(
                                    paper_bgcolor="#1e1e2f",
                                    plot_bgcolor="#1e1e2f",
                                    margin=dict(r=20, t=40, l=20, b=20)
                                )
                            ),
                        ]
                    ),
                ]
            ),
        ]
    elif pathname == '/customer-segmentation':
        return [
            # kmeans visualization
            html.Div(
                className="card card-large",
                id="clustering-3d",
                children=[
                    html.H3("3D Clustering Visualization"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.scatter_3d(
                                rfm_combined, x='Recency', y='Frequency', z='Monetary',
                                color='Cluster', title="3D Clustering Visualization", template="plotly_dark"
                            ).update_layout(
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),

            html.Div(
                className="card card-large",
                id="clustering-bar",
                children=[
                    html.H3("Cluster Sizes"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=px.bar(
                                cluster_summary, x=cluster_summary.index, y='Cluster_size',
                                title="Cluster Sizes", template="plotly_dark"
                            ).update_layout(
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),
            html.Div(
                className="card",
                id="recency-metric",
                children=[
                    html.H3("Recency Metric"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=recency_fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),

            html.Div(
                className="card",
                id="frequency-metric",
                children=[
                    html.H3("Frequency Metric"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=frequency_fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),

            html.Div(
                className="card",
                id="monetary-metric",
                children=[
                    html.H3("Monetary Metric"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=monetary_fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),

        ]
    else:  # General Analytics (default page)
        return [
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
                                template="plotly_dark"
                            )),
                        ]
                    )
                ]
            ),

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
                                template="plotly_dark"
                            )),
                        ]
                    )
                ]
            ),

            html.Div(
                className="card",
                id="return-rate",
                children=[
                    html.H3("Return Rate Analysis"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=return_rate_chart.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),

            html.Div(
                className="card",
                id="returned-products",
                children=[
                    html.H3("Most Returned Products"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=returned_products_chart.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),

            html.Div(
                className="card card-sales-metrics",
                id="sales-metrics",
                children=[
                    html.H3("Average Sales Metrics"),
                    html.Div(
                        className="sales-metrics-container",
                        children=[
                            html.Div(
                                className="metric-card",
                                children=[
                                    html.P("Average Daily Sales"),
                                    html.H4(f"${avg_day:.2f}")
                                ]
                            ),
                            html.Div(
                                className="metric-card",
                                children=[
                                    html.P("Average Weekly Sales"),
                                    html.H4(f"${avg_week:.2f}")
                                ]
                            ),
                            html.Div(
                                className="metric-card",
                                children=[
                                    html.P("Average Monthly Sales"),
                                    html.H4(f"${avg_month:.2f}")
                                ]
                            ),
                        ]
                    ),
                ]
            ),

            html.Div(
                className="card",
                id="top-hours",
                children=[
                    html.H3("Top Performing Hours"),
                    html.Div(
                        className="card-graph",
                        children=[
                            dcc.Graph(figure=hourly_sales_chart.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#1e1e2f",
                                plot_bgcolor="#1e1e2f",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                        ]
                    )
                ]
            ),
        ]


# Layout with URL Routing
app.layout = html.Div([
    nav_panel,
    dcc.Location(id='url', refresh=False),  # Track URL
    html.Div(id='page-content', className="content"),  # Dynamic page content
])

if __name__ == '__main__':
    app.run_server(debug=True)
