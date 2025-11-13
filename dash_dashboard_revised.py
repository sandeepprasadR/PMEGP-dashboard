"""
PMEGP Interactive Dashboard - Dash Implementation (REVISED)
============================================================
This script creates an interactive Dash dashboard with automatic
CSV loading from the KPI Dashboard folder and manual upload option.

Features:
  - Auto-load CSV from default path
  - Manual file upload as backup
  - Interactive filters with real-time updates
  - Professional dashboard design
  - Multiple tabs for exploration
  - Data export capability
  - Responsive layout

Author: Generated for MoMSME Executive Dashboard
Date: November 2025
Purpose: Interactive executive dashboard for PMEGP monitoring
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import base64
import io

warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_prepare_data(csv_path):
    """Load and prepare PMEGP data"""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Convert date columns
        date_cols = ['application_date', 'sanction_date', 'disbursement_date', 
                     'last_inspection_date', 'second_loan_date', 'upgrade_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Feature engineering
        df['processing_time_days'] = (df['disbursement_date'] - df['application_date']).dt.days
        df['turnover_to_cost_ratio'] = df['annual_turnover_rs'] / (df['project_cost_rs'] + 1)
        df['employment_efficiency'] = df['employment_at_setup'] / (df['project_cost_rs'] / 100000 + 1)
        df['roi_percent'] = ((df['annual_turnover_rs'] - df['project_cost_rs']) / 
                            df['project_cost_rs'] * 100).fillna(0)
        df['month_year'] = df['application_date'].dt.strftime('%Y-%m')
        df['year'] = df['application_date'].dt.year
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def parse_upload(contents, filename):
    """Parse uploaded CSV file"""
    if contents is None:
        return None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Feature engineering
        date_cols = ['application_date', 'sanction_date', 'disbursement_date', 
                     'last_inspection_date', 'second_loan_date', 'upgrade_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df['roi_percent'] = ((df['annual_turnover_rs'] - df['project_cost_rs']) / 
                            df['project_cost_rs'] * 100).fillna(0)
        df['month_year'] = df['application_date'].dt.strftime('%Y-%m')
        
        return df
    except Exception as e:
        print(f"Error parsing upload: {e}")
        return None

# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

def create_state_heatmap(df):
    """Create state performance heatmap"""
    if df is None or len(df) == 0:
        return go.Figure().add_annotation(text="No data available")
    
    state_metrics = df.groupby('state').agg({
        'enterprise_id': 'count',
        'employment_at_setup': 'sum',
        'margin_money_subsidy_rs': 'sum',
        'operational_status': lambda x: (x == 'Operational').sum()
    }).reset_index()
    
    state_metrics['Operational_Rate'] = (state_metrics['operational_status'] / state_metrics['enterprise_id'] * 100)
    state_metrics = state_metrics.sort_values('employment_at_setup', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=state_metrics['state'],
        x=state_metrics['employment_at_setup'],
        name='Employment',
        orientation='h',
        marker=dict(
            color=state_metrics['margin_money_subsidy_rs'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Subsidy (Rs)")
        ),
        customdata=np.stack((
            state_metrics['enterprise_id'],
            state_metrics['Operational_Rate']
        ), axis=-1),
        hovertemplate='<b>%{y}</b><br>' +
                     'Employment: %{x:,.0f}<br>' +
                     'Enterprises: %{customdata[0]:.0f}<br>' +
                     'Operational: %{customdata[1]:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='State Performance Dashboard',
        xaxis_title='Employment Generated',
        yaxis_title='State',
        height=600,
        template='plotly_white',
        hovermode='y unified'
    )
    
    return fig

def create_sector_sunburst(df):
    """Create sunburst chart"""
    if df is None or len(df) == 0:
        return go.Figure().add_annotation(text="No data available")
    
    hierarchy_data = df.groupby(['sector', 'location_type']).agg({
        'employment_at_setup': 'sum',
        'enterprise_id': 'count'
    }).reset_index()
    
    fig = go.Figure()
    
    sectors = hierarchy_data['sector'].unique()
    for sector in sectors:
        sector_data = hierarchy_data[hierarchy_data['sector'] == sector]
        fig.add_trace(go.Bar(
            x=sector_data['location_type'],
            y=sector_data['employment_at_setup'],
            name=sector,
            hovertemplate='<b>%{x}</b><br>Employment: %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Employment by Sector and Location',
        xaxis_title='Location Type',
        yaxis_title='Employment',
        height=500,
        barmode='group',
        template='plotly_white'
    )
    
    return fig

def create_time_series_chart(df):
    """Create time series chart"""
    if df is None or len(df) == 0:
        return go.Figure().add_annotation(text="No data available")
    
    monthly_data = df.groupby('month_year').agg({
        'enterprise_id': 'count',
        'employment_at_setup': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Enterprise Setup', 'Employment Generated')
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month_year'],
            y=monthly_data['enterprise_id'],
            name='Enterprises',
            mode='lines+markers',
            line=dict(color='#2E86AB', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month_year'],
            y=monthly_data['employment_at_setup'],
            name='Employment',
            mode='lines+markers',
            line=dict(color='#A23B72', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, template='plotly_white', hovermode='x unified')
    return fig

def create_demographic_charts(df):
    """Create demographic analysis"""
    if df is None or len(df) == 0:
        return go.Figure().add_annotation(text="No data available")
    
    gender_data = df['gender'].value_counts().reset_index()
    category_data = df['category'].value_counts().reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=("Gender Distribution", "Social Category Distribution")
    )
    
    fig.add_trace(
        go.Pie(labels=gender_data['gender'], values=gender_data['count'],
               marker=dict(colors=['#FF6B6B', '#4ECDC4']), name='Gender'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(labels=category_data['category'], values=category_data['count'],
               marker=dict(colors=['#FFE66D', '#95E1D3', '#F38181', '#AA96DA']),
               name='Category'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, template='plotly_white')
    return fig

def create_financial_chart(df):
    """Create financial analysis"""
    if df is None or len(df) == 0:
        return go.Figure().add_annotation(text="No data available")
    
    sector_financial = df.groupby('sector').agg({
        'project_cost_rs': 'mean',
        'margin_money_subsidy_rs': 'mean',
        'annual_turnover_rs': 'mean'
    }) / 100000
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sector_financial.index,
        y=sector_financial['project_cost_rs'],
        name='Project Cost',
        marker=dict(color='#FF6B6B')
    ))
    
    fig.add_trace(go.Bar(
        x=sector_financial.index,
        y=sector_financial['margin_money_subsidy_rs'],
        name='Subsidy',
        marker=dict(color='#4ECDC4')
    ))
    
    fig.add_trace(go.Bar(
        x=sector_financial.index,
        y=sector_financial['annual_turnover_rs'],
        name='Turnover',
        marker=dict(color='#06A77D')
    ))
    
    fig.update_layout(
        title='Financial Metrics by Sector (Lakhs)',
        xaxis_title='Sector',
        yaxis_title='Amount (Lakhs)',
        barmode='group',
        height=500,
        template='plotly_white'
    )
    
    return fig

# ============================================================================
# INITIALIZE DASH APP
# ============================================================================

app = dash.Dash(__name__)

# Try to load default data
csv_path = '/Users/apple/Documents/0. MSME project/KPI Dashboard/PMEGP_KPI_Dataset_2023-26.csv'
default_df = load_and_prepare_data(csv_path)

# ============================================================================
# APP LAYOUT
# ============================================================================

app.layout = html.Div([
    # Store component for data
    dcc.Store(id='data-store', data=None),
    
    # Header
    html.Div([
        html.H1("PMEGP Executive Dashboard", style={'textAlign': 'center', 'color': '#1f77b4', 'marginBottom': 10}),
        html.P("Real-time Monitoring of Prime Minister's Employment Generation Programme",
               style={'textAlign': 'center', 'color': '#666', 'fontSize': 14})
    ], style={'padding': '30px 20px', 'backgroundColor': '#f8f9fa', 'borderBottom': '2px solid #1f77b4'}),
    
    # Data loading section
    html.Div([
        html.Div([
            html.H3("Data Source"),
            html.Div(id='data-status', style={'padding': '10px', 'marginBottom': '10px'}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                }
            )
        ], style={'backgroundColor': '#f0f2f6', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'})
    ], style={'padding': '20px'}),
    
    # Filters Row
    html.Div([
        html.Div([
            html.Label("Select State(s):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='state-filter', multi=True, style={'width': '100%'})
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Select Sector(s):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='sector-filter', multi=True, style={'width': '100%'})
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Select Category:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='category-filter', multi=True, style={'width': '100%'})
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Select Status:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='status-filter', multi=True, style={'width': '100%'})
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'margin': '20px', 'borderRadius': '10px'}),
    
    # KPI Metrics Row
    html.Div(id='metrics-row', style={'padding': '20px', 'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}),
    
    # Tabs
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Overview', value='tab-1', children=[
            html.Div([
                dcc.Graph(id='state-heatmap'),
                dcc.Graph(id='time-series-chart')
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='Sector Analysis', value='tab-2', children=[
            html.Div([
                dcc.Graph(id='sunburst-chart'),
                dcc.Graph(id='financial-chart')
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='Demographics', value='tab-3', children=[
            html.Div([
                dcc.Graph(id='demographic-chart')
            ], style={'padding': '20px'})
        ]),
        
        dcc.Tab(label='Data', value='tab-4', children=[
            html.Div([
                html.Div(id='data-summary', style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px'}),
                html.Div(id='data-table', style={'padding': '20px'})
            ])
        ])
    ], style={'margin': '20px'}),
    
    # Footer
    html.Footer(
        'PMEGP Executive Dashboard | Ministry of MSME | Data as of Nov 12, 2025',
        style={'textAlign': 'center', 'padding': '20px', 'color': '#888', 'fontSize': 12, 'borderTop': '1px solid #ddd', 'marginTop': '40px'}
    )
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f5f5', 'minHeight': '100vh'})

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('data-store', 'data'),
     Output('data-status', 'children'),
     Output('state-filter', 'options'),
     Output('state-filter', 'value'),
     Output('sector-filter', 'options'),
     Output('sector-filter', 'value'),
     Output('category-filter', 'options'),
     Output('category-filter', 'value'),
     Output('status-filter', 'options'),
     Output('status-filter', 'value')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def load_data(contents, filename):
    """Load data from upload or default path"""
    
    df = None
    status = "❌ No data loaded"
    
    # Try upload first
    if contents is not None:
        df = parse_upload(contents, filename)
        if df is not None:
            status = f"✓ Loaded from upload: {filename}"
    
    # Fall back to default
    if df is None and default_df is not None:
        df = default_df
        status = "✓ Loaded from default path"
    
    if df is None:
        return None, status, [], [], [], [], [], [], [], []
    
    # Prepare dropdown options
    states = sorted(df['state'].unique())
    sectors = sorted(df['sector'].unique())
    categories = sorted(df['category'].unique())
    statuses = sorted(df['operational_status'].unique())
    
    return df.to_json(date_format='iso', orient='split'), status, \
           [{'label': s, 'value': s} for s in states], states, \
           [{'label': s, 'value': s} for s in sectors], sectors, \
           [{'label': c, 'value': c} for c in categories], categories, \
           [{'label': s, 'value': s} for s in statuses], statuses

@app.callback(
    [Output('metrics-row', 'children'),
     Output('state-heatmap', 'figure'),
     Output('time-series-chart', 'figure'),
     Output('sunburst-chart', 'figure'),
     Output('financial-chart', 'figure'),
     Output('demographic-chart', 'figure'),
     Output('data-summary', 'children'),
     Output('data-table', 'children')],
    [Input('data-store', 'data'),
     Input('state-filter', 'value'),
     Input('sector-filter', 'value'),
     Input('category-filter', 'value'),
     Input('status-filter', 'value')]
)
def update_dashboard(data_json, states, sectors, categories, statuses):
    """Update all dashboard components"""
    
    if data_json is None:
        empty_fig = go.Figure().add_annotation(text="Load data first")
        return ([], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, 
                html.P("No data loaded"), html.P("No data loaded"))
    
    df = pd.read_json(data_json, orient='split')
    
    # Apply filters
    filtered_df = df.copy()
    if states:
        filtered_df = filtered_df[filtered_df['state'].isin(states)]
    if sectors:
        filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if statuses:
        filtered_df = filtered_df[filtered_df['operational_status'].isin(statuses)]
    
    # Metrics cards
    metrics_cards = [
        html.Div([
            html.H3(f"{len(filtered_df):,}"),
            html.P("Enterprises")
        ], style={'backgroundColor': '#e3f2fd', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'flex': 1, 'minWidth': '150px'}),
        
        html.Div([
            html.H3(f"{int(filtered_df['employment_at_setup'].sum()):,}"),
            html.P("Employment")
        ], style={'backgroundColor': '#f3e5f5', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'flex': 1, 'minWidth': '150px'}),
        
        html.Div([
            html.H3(f"₹{round(filtered_df['margin_money_subsidy_rs'].sum() / 10000000, 2)}Cr"),
            html.P("Subsidy")
        ], style={'backgroundColor': '#fff3e0', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'flex': 1, 'minWidth': '150px'}),
        
        html.Div([
            html.H3(f"{round((filtered_df['operational_status'] == 'Operational').sum() / len(filtered_df) * 100, 1)}%"),
            html.P("Operational")
        ], style={'backgroundColor': '#e8f5e9', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'flex': 1, 'minWidth': '150px'}),
    ]
    
    # Create charts
    state_heatmap = create_state_heatmap(filtered_df)
    time_series = create_time_series_chart(filtered_df)
    sunburst = create_sector_sunburst(filtered_df)
    financial = create_financial_chart(filtered_df)
    demographic = create_demographic_charts(filtered_df)
    
    # Data summary
    summary = html.Div([
        html.H3(f"Showing {len(filtered_df)} of {len(df)} enterprises"),
        html.P(f"Female: {round((filtered_df['gender'] == 'Female').sum() / len(filtered_df) * 100, 1)}% | " +
               f"SC/ST: {round(len(filtered_df[filtered_df['category'].isin(['SC', 'ST'])]) / len(filtered_df) * 100, 1)}% | " +
               f"Rural: {round((filtered_df['location_type'] == 'Rural').sum() / len(filtered_df) * 100, 1)}%")
    ])
    
    # Data table
    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Enterprise ID'),
                html.Th('State'),
                html.Th('Sector'),
                html.Th('Employment'),
                html.Th('Status')
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(row['enterprise_id']),
                html.Td(row['state']),
                html.Td(row['sector']),
                html.Td(f"{row['employment_at_setup']}"),
                html.Td(row['operational_status'])
            ]) for row in filtered_df.head(30).to_dict('records')
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse'})
    
    return metrics_cards, state_heatmap, time_series, sunburst, financial, demographic, summary, table

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
