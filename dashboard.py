import dash
from dash import dcc, html, Input, Output
from main import *
import plotly.express as px
import os

# Load and analyze data
df = load_and_prepare_data()
rfm_dataset, rfm_normalized = calculate_rfm(df)
rfm_combined, rfm_normalized = cluster_rfm(rfm_normalized, rfm_dataset)
cluster_summary = cluster_breakdown(rfm_combined)
time_series, trend, seasonal, residual = time_series_analysis(df)
sales_country = sales_by_country(df)
top_products_data = top_products(df)
recency_fig, frequency_fig, monetary_fig = cluster_statistics_visualization(cluster_summary)
return_rate_chart = calculate_return_rate(df)
returned_products_chart = most_returned_products(df)
avg_day, avg_week, avg_month = average_sales_by_time(df)
hourly_sales_chart = top_performing_hours(df)
cluster_summary_reset = cluster_summary.reset_index()
cluster_summary_reset['index'] = cluster_summary_reset['index'].astype(str)  # Convert to string for discrete colors
rfm_combined['Cluster'] = rfm_combined['Cluster'].astype(str)
top_products_data['Truncated_Description'] = top_products_data['Description'].apply(
    lambda x: x[:15] + "..." if len(x) > 15 else x
)
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
# Metrics Section and Divider (for General Analytics)
metrics_section = html.Div(
    id="metrics-section",
    className="metrics-container",
    children=[
        html.Div(
            className="metric-card",
            children=[
                html.P("Average Daily Sales", className="metric-label"),
                html.H4(f"${avg_day:.2f}", className="metric-value"),
            ]
        ),
        html.Div(
            className="metric-card",
            children=[
                html.P("Average Weekly Sales", className="metric-label"),
                html.H4(f"${avg_week:.2f}", className="metric-value"),
            ]
        ),
        html.Div(
            className="metric-card",
            children=[
                html.P("Average Monthly Sales", className="metric-label"),
                html.H4(f"${avg_month:.2f}", className="metric-value"),
            ]
        ),
    ],
    style={'display': 'none'}  # Default to hidden
)

divider = html.Div(
    id="divider-line",
    className="divider-line",
    style={'display': 'none'}  # Default to hidden
)
# Page Content
@app.callback(
    [Output('page-content', 'children'),
     Output('metrics-section', 'style'),
     Output('divider-line', 'style')],
    Input('url', 'pathname'))
def render_page_content(pathname):
    # Time Series graph
    if pathname == '/time-series':
        return [
            [
                html.Div(
                    className="plotly-graph card-large",  # Apply the custom CSS class
                    id="time-series-raw",
                    children=[
                                dcc.Graph(
                                    figure=px.line(
                                        x=time_series.index,
                                        y=time_series.values,
                                        template="plotly_dark"
                                    ).update_layout(
                                        title={
                                            'text': "Raw Time Series",
                                            'x': 0.5,
                                            'xanchor': 'center',
                                            'yanchor': 'top'
                                        },
                                        paper_bgcolor="#1d2025",
                                        plot_bgcolor="#1d2025",
                                        margin=dict(r=20, t=40, l=20, b=20)
                                    ),
                                    style={'width': '100%', 'height': '100%'}
                                ),
                    ]
                ),
                html.Div(
                    className="plotly-graph card-large",
                    id="time-series-trend",
                    children=[
                                dcc.Graph(
                                    figure=px.line(
                                        x=trend.index,
                                        y=trend.values,
                                        template="plotly_dark"
                                    ).update_layout(
                                        title={
                                            'text': "Time Series Trend",
                                            'x': 0.5,
                                            'xanchor': 'center',
                                            'yanchor': 'top'
                                        },
                                        paper_bgcolor="#1d2025",
                                        plot_bgcolor="#1d2025",
                                        margin=dict(r=20, t=40, l=20, b=20)
                                    ),
                                    style={'width': '100%', 'height': '100%'}
                                ),
                    ]
                ),
            ],  # Content for 'page-content'
            {'display': 'none'},  # Style for 'metrics-section'
            {'display': 'none'},  # Style for 'divider-line'
        ]
    elif pathname == '/customer-segmentation':
        return [
            [
            # kmeans visualization
            html.Div(
                className="plotly-graph",
                id="clustering-3d",
                children=[
                            dcc.Graph(figure=px.scatter_3d(
                                rfm_combined, x='Recency', y='Frequency', z='Monetary',
                                color='Cluster',  # The 'Cluster' column contains [1, 0, 2, 3]
                                color_discrete_sequence=[
                                     '#A7C7E7',
                                     '#6082B6',
                                     '#87CEEB',
                                     '#4682B4'
                                ],
                                title="3D Clustering Visualization", template="plotly_dark"
                            ).update_layout(
                                title={
                                    'text': "3D Clustering Visualization",  # Title text
                                    'x': 0.5,  # Center the title
                                    'xanchor': 'center',  # Anchor the title to the center
                                    'yanchor': 'top'  # Anchor to the top of the graph
                                },
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=40, l=20, b=20)
                            ),
                                style={'width': '100%', 'height': '100%'}  # Fill parent container
                            ),
                ]
            ),

            html.Div(
                className="plotly-graph card-large",
                id="clustering-bar",
                children=[
                            dcc.Graph(figure=px.bar(
                                cluster_summary_reset, x='index', y='Cluster_size',
                                color='index',
                                color_discrete_sequence=[  # Explicitly map cluster IDs to custom colors
                                     '#A7C7E7',
                                     '#6082B6',
                                     '#87CEEB',
                                     '#4682B4'
                                ],
                                title="Cluster Sizes", template="plotly_dark"
                            ).update_layout(
                                title={
                                    'text': "Cluster Sizes",  # Title text
                                    'x': 0.5,  # Center the title
                                    'xanchor': 'center',  # Anchor the title to the center
                                    'yanchor': 'top'  # Anchor to the top of the graph
                                },
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=40, l=20, b=20)
                            ),
                                style={'width': '100%', 'height': '100%'}  # Fill parent container
                            ),
                ]
            ),
            html.Div(
                className="plotly-graph",
                id="recency-metric",
                children=[
                        dcc.Graph(figure=recency_fig.update_layout(
                            title={
                                'text': "Recency by Segment",  # Title text
                                'x': 0.5,  # Center the title
                                'xanchor': 'center',  # Anchor the title to the center
                                'yanchor': 'top'  # Anchor to the top of the graph
                            },
                            template="plotly_dark",
                            paper_bgcolor="#1d2025",
                            plot_bgcolor="#1d2025",
                            margin=dict(r=20, t=40, l=20, b=20)
                        )),
            ]
        ),

        html.Div(
                className="plotly-graph",
                id="frequency-metric",
                children=[
                            dcc.Graph(figure=frequency_fig.update_layout(
                                title={
                                    'text': "Frequency by Segment",  # Title text
                                    'x': 0.5,  # Center the title
                                    'xanchor': 'center',  # Anchor the title to the center
                                    'yanchor': 'top'  # Anchor to the top of the graph
                                },
                                template="plotly_dark",
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                ]
            ),

            html.Div(
                className="plotly-graph",
                id="monetary-metric",
                children=[
                            dcc.Graph(figure=monetary_fig.update_layout(
                                title={
                                    'text': "Monetary by Segment",  # Title text
                                    'x': 0.5,  # Center the title
                                    'xanchor': 'center',  # Anchor the title to the center
                                    'yanchor': 'top'  # Anchor to the top of the graph
                                },
                                template="plotly_dark",
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                ]
            ),
            ],
            {'display': 'none'},  # Hide metrics
            {'display': 'none'},  # Hide divider
        ]
    else:  # General Analytics (default page)

        return [
            [
            html.Div(
                className="plotly-graph",
                id="sales-country",
                children=[
                            dcc.Graph(figure=px.choropleth(
                                sales_country, locations="Country",
                                locationmode="country names", color="Revenue",
                                template="plotly_dark",
                            ).update_layout(
                                title={
                                    'text': "Sales by Country",  # Title text
                                    'x': 0.5,  # Center the title
                                    'xanchor': 'center',  # Anchor the title to the center
                                    'yanchor': 'top'  # Anchor to the top of the graph
                                },
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=40, l=20, b=20)
                            ),
                            ),
                ]
            ),

            html.Div(
                className="plotly-graph",
                id="top-products",
                children=[
                            dcc.Graph(figure=px.bar(
                                top_products_data, x='Truncated_Description', y='Revenue',
                                template="plotly_dark"
                            ).update_traces(marker_color="#6082B6").update_layout(
                                title={
                                    'text': "Top Products by Revenue",  # Title text
                                    'x': 0.5,  # Center the title
                                    'xanchor': 'center',  # Anchor the title to the center
                                    'yanchor': 'top'  # Anchor to the top of the graph
                                },
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=40, l=20, b=20)
                            )),
                ]
            ),

            html.Div(
                className="plotly-graph",
                id="return-rate",
                children=[
                            dcc.Graph(figure=return_rate_chart.update_layout(
                                title={
                                    'text': "Return Rate Analysis",  # Title text
                                    'x': 0.5,  # Center alignment
                                    'xanchor': 'center',  # Anchor title to the center
                                    'yanchor': 'top',  # Anchor title to the top
                                },
                                template="plotly_dark",
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=80, l=20, b=40),
                                showlegend=False,  # Show legend (optional)

                            ),
                            ),
                ]
            ),
            html.Div(
                className="plotly-graph",
                id="returned-products",
                children=[
                            dcc.Graph(figure=returned_products_chart.update_layout(
                                title={
                                    'text': "Most Returned Products",  # Title text
                                    'x': 0.5,  # Center alignment
                                    'xanchor': 'center',  # Anchor title to the center
                                    'yanchor': 'top',  # Anchor title to the top
                                },
                                template="plotly_dark",
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                autosize=True,
                                margin=dict(r=20, t=40, l=20, b=20)
                            ),
                                style={'width': '100%', 'height': '100%'}  # Fill parent container
                            ),
                ]
            ),

            html.Div(
                className="plotly-graph",
                id="top-hours",
                children=[
                            dcc.Graph(figure=hourly_sales_chart.update_layout(
                                title={
                                    'text': "Top Performing Hours",  # Title text
                                    'x': 0.5,  # Center alignment
                                    'xanchor': 'center',  # Anchor title to the center
                                    'yanchor': 'top',  # Anchor title to the top
                                },
                                template="plotly_dark",
                                paper_bgcolor="#1d2025",
                                plot_bgcolor="#1d2025",
                                margin=dict(r=20, t=40, l=20, b=20)
                            ),
                                style={'width': '100%', 'height': '100%'}  # Fill parent container
                            ),
                ]
            ),

            ],
            {'display': 'flex'},  # Show metrics
            {'display': 'block'},  # Show divider
        ]


# Layout with URL Routing
app.layout = html.Div([
    nav_panel,  # Sidebar navigation
    dcc.Location(id='url', refresh=False),  # Track URL
    metrics_section,
    divider,
    html.Div(id='page-content', className="content"),  # Dynamic page content
])


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Default to 8050 for local testing
    app.run_server(debug=True, port=port, host='0.0.0.0')

server = app.server
