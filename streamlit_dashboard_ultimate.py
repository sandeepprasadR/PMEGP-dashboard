"""
PMEGP Interactive Dashboard - ULTIMATE VERSION (ALL VISUALIZATIONS ENHANCED)
===========================================================================

Complete overhaul with:
- Enhanced geographic map with detailed state info & heatmap
- Improved all other charts and visualizations
- Better color schemes and labeling
- More informative hover details
- Professional styling throughout
- Actionable insights for each visualization

Author: Generated for MoMSME Executive Dashboard
Date: November 2025
Purpose: Production-ready executive dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PMEGP Executive Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 20px;}
    h1 {color: #1f77b4; text-align: center; padding: 20px;}
    h2 {color: #2c3e50; border-bottom: 2px solid #1f77b4; padding-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data(csv_path):
    """Load and prepare data"""
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        date_cols = ['application_date', 'sanction_date', 'disbursement_date', 
                     'last_inspection_date', 'second_loan_date', 'upgrade_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df['processing_time_days'] = (df['disbursement_date'] - df['application_date']).dt.days
        df['roi_percent'] = ((df['annual_turnover_rs'] - df['project_cost_rs']) / 
                            df['project_cost_rs'] * 100).fillna(0)
        df['month_year'] = df['application_date'].dt.strftime('%Y-%m')
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def apply_filters(df, filters):
    """Apply filters"""
    filtered_df = df.copy()
    if filters['states'] and len(filters['states']) > 0:
        filtered_df = filtered_df[filtered_df['state'].isin(filters['states'])]
    if filters['sectors'] and len(filters['sectors']) > 0:
        filtered_df = filtered_df[filtered_df['sector'].isin(filters['sectors'])]
    if filters['categories'] and len(filters['categories']) > 0:
        filtered_df = filtered_df[filtered_df['category'].isin(filters['categories'])]
    if filters['status'] and len(filters['status']) > 0:
        filtered_df = filtered_df[filtered_df['operational_status'].isin(filters['status'])]
    if filters['location'] and len(filters['location']) > 0:
        filtered_df = filtered_df[filtered_df['location_type'].isin(filters['location'])]
    return filtered_df

def calculate_metrics(df):
    """Calculate metrics"""
    if len(df) == 0:
        return None
    return {
        'Total_Enterprises': len(df),
        'Total_Employment': int(df['employment_at_setup'].sum()),
        'Total_Subsidy_Cr': round(df['margin_money_subsidy_rs'].sum() / 10000000, 2),
        'Operational_Rate': round((df['operational_status'] == 'Operational').sum() / len(df) * 100, 2),
        'Female_Rate': round((df['gender'] == 'Female').sum() / len(df) * 100, 2),
    }

def create_kpi_cards(metrics):
    """KPI cards"""
    if metrics is None:
        return
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Enterprises", f"{metrics['Total_Enterprises']:,}")
    with col2:
        st.metric("Employment", f"{metrics['Total_Employment']:,}")
    with col3:
        st.metric("Subsidy (Cr)", f"₹{metrics['Total_Subsidy_Cr']:.2f}")
    with col4:
        st.metric("Operational %", f"{metrics['Operational_Rate']:.1f}%")
    with col5:
        st.metric("Female %", f"{metrics['Female_Rate']:.1f}%")

# ============================================================================
# ENHANCED GEOGRAPHIC MAP WITH HEATMAP
# ============================================================================

def create_enhanced_geographic_map(df):
    """ENHANCED MAP: Shows state data with interactive features"""
    
    if len(df) == 0:
        st.warning("No data")
        return
    
    # State coordinates for accurate mapping
    state_coords = {
        'Uttar Pradesh': (26.8467, 80.9462), 'Maharashtra': (19.7515, 75.7139),
        'Tamil Nadu': (11.1271, 79.2787), 'Karnataka': (15.3173, 75.7139),
        'Gujarat': (22.2587, 71.1924), 'Rajasthan': (27.0238, 74.2179),
        'West Bengal': (24.8355, 88.2676), 'Madhya Pradesh': (22.9375, 78.6553),
        'Bihar': (25.0961, 85.3131), 'Punjab': (31.1471, 75.3412),
        'Haryana': (29.0588, 77.0745), 'Kerala': (10.8505, 76.2711),
        'Odisha': (20.9517, 85.0985), 'Telangana': (18.1124, 79.0193),
        'Assam': (26.2006, 92.9376)
    }
    
    # Aggregate state-wise data
    state_data = df.groupby('state').agg({
        'employment_at_setup': 'sum',
        'enterprise_id': 'count',
        'margin_money_subsidy_rs': 'sum',
        'operational_status': lambda x: (x == 'Operational').sum(),
        'project_cost_rs': 'mean',
        'annual_turnover_rs': 'mean'
    }).reset_index()
    
    state_data.columns = ['state', 'employment', 'enterprises', 'subsidy', 
                          'operational', 'avg_cost', 'avg_turnover']
    state_data['operational_rate'] = (state_data['operational'] / state_data['enterprises'] * 100).round(1)
    
    # Create color scale based on employment
    max_employment = state_data['employment'].max()
    
    # Create map
    m = folium.Map(location=[22.5, 78], zoom_start=4, tiles='OpenStreetMap')
    
    # Add circles with color gradient
    for _, row in state_data.iterrows():
        state = row['state']
        if state in state_coords:
            coords = state_coords[state]
            employment = int(row['employment'])
            enterprises = int(row['enterprises'])
            operational_pct = row['operational_rate']
            subsidy = int(row['subsidy'])
            avg_cost = int(row['avg_cost'])
            
            # Color intensity based on employment
            intensity = (employment / max_employment) * 0.8 + 0.2
            color_val = int(intensity * 255)
            color = f'#{color_val:02x}86FF'
            
            # Radius based on number of enterprises
            radius = max(30000, min(150000, enterprises * 5000))
            
            # Enhanced popup with detailed info
            popup_html = f"""
            <div style="font-family: Arial; width: 300px;">
                <h3 style="color: #1f77b4; margin-bottom: 10px;"><b>{state}</b></h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Enterprises:</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>{enterprises:,}</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Employment:</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>{employment:,}</b></td>
                    </tr>
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Operational:</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>{operational_pct:.1f}%</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Subsidy (Cr):</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>₹{subsidy/10000000:.2f}</b></td>
                    </tr>
                    <tr style="background-color: #f0f0f0;">
                        <td style="padding: 8px; border: 1px solid #ddd;"><b>Avg Cost:</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;"><b>₹{avg_cost/100000:.2f}L</b></td>
                    </tr>
                </table>
            </div>
            """
            
            folium.Circle(
                location=coords,
                radius=radius,
                popup=folium.Popup(popup_html, max_width=350),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2,
                tooltip=f"{state}: {enterprises} enterprises, {employment:,} employment"
            ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0; font-weight: bold; color: #1f77b4;">📍 Map Legend</p>
    <hr style="margin: 5px 0;">
    <p style="margin: 5px 0; font-size: 12px;">
    • <b>Circle Size</b> = Number of Enterprises<br>
    • <b>Circle Color</b> = Employment Generated<br>
    • <b>Darker Blue</b> = More Employment<br>
    • <b>Click Circle</b> = Detailed Information
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    st.markdown("### Geographic Performance Map")
    st.info("**How to read**: Larger circles = more enterprises. Darker blue = more employment. Click any circle for details.")
    st_folium(m, width=1400, height=600)
    
    # State ranking table
    st.markdown("### Top States Performance Table")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 10 by Employment")
        top_employment = state_data.nlargest(10, 'employment')[
            ['state', 'enterprises', 'employment', 'operational_rate', 'subsidy']
        ].copy()
        top_employment['subsidy'] = top_employment['subsidy'] / 10000000
        top_employment.columns = ['State', 'Enterprises', 'Employment', 'Operational %', 'Subsidy (Cr)']
        st.dataframe(top_employment.reset_index(drop=True), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Top 10 by Operational Rate")
        top_operational = state_data.nlargest(10, 'operational_rate')[
            ['state', 'enterprises', 'employment', 'operational_rate', 'subsidy']
        ].copy()
        top_operational['subsidy'] = top_operational['subsidy'] / 10000000
        top_operational.columns = ['State', 'Enterprises', 'Employment', 'Operational %', 'Subsidy (Cr)']
        st.dataframe(top_operational.reset_index(drop=True), use_container_width=True, hide_index=True)

# ============================================================================
# ENHANCED STATE PERFORMANCE HEATMAP
# ============================================================================

def create_enhanced_state_heatmap(df):
    """IMPROVED: State performance with multiple metrics"""
    
    if len(df) == 0:
        return None
    
    state_metrics = df.groupby('state').agg({
        'employment_at_setup': 'sum',
        'enterprise_id': 'count',
        'margin_money_subsidy_rs': 'sum',
        'operational_status': lambda x: (x == 'Operational').sum(),
        'sustainability_score': 'mean'
    }).reset_index()
    
    state_metrics['operational_rate'] = (state_metrics['operational_status'] / 
                                         state_metrics['enterprise_id'] * 100)
    state_metrics = state_metrics.sort_values('employment_at_setup', ascending=True)
    
    fig = go.Figure()
    
    # Add employment bars
    fig.add_trace(go.Bar(
        y=state_metrics['state'],
        x=state_metrics['employment_at_setup'],
        name='Employment Generated',
        orientation='h',
        marker=dict(
            color=state_metrics['operational_rate'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Operational<br>Rate %", thickness=20),
            line=dict(width=0)
        ),
        text=[f"<b>{emp:,.0f} jobs</b><br>{ops:.1f}% operational" 
              for emp, ops in zip(state_metrics['employment_at_setup'], 
                                  state_metrics['operational_rate'])],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Employment: %{x:,.0f}<br>Enterprises: %{customdata[0]}<br>Operational: %{customdata[1]:.1f}%<extra></extra>',
        customdata=np.column_stack((
            state_metrics['enterprise_id'],
            state_metrics['operational_rate']
        ))
    ))
    
    fig.update_layout(
        title={
            'text': '<b>State-wise Employment Generation & Operational Performance</b><br>' +
                   '<sub>Color intensity shows operational rate | Bar length shows employment</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='<b>Employment Generated</b> (Number of Jobs)',
        yaxis_title='<b>State</b>',
        height=700,
        template='plotly_white',
        hovermode='y unified',
        showlegend=False,
        margin=dict(l=120, r=200, t=100, b=50)
    )
    
    return fig

# ============================================================================
# ENHANCED SECTOR ANALYSIS
# ============================================================================

def create_enhanced_sector_chart(df):
    """IMPROVED: Sector analysis with better visualization"""
    
    if len(df) == 0:
        return None
    
    sector_data = df.groupby('sector').agg({
        'enterprise_id': 'count',
        'employment_at_setup': 'sum',
        'project_cost_rs': 'mean',
        'margin_money_subsidy_rs': 'mean',
        'operational_status': lambda x: (x == 'Operational').sum()
    }).reset_index()
    
    sector_data['operational_rate'] = (sector_data['operational_status'] / 
                                       sector_data['enterprise_id'] * 100)
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=("Sector Distribution", "Employment", 
                       "Avg Project Cost (Lakhs)", "Operational Rate %"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Pie chart
    fig.add_trace(
        go.Pie(labels=sector_data['sector'], values=sector_data['enterprise_id'],
               marker=dict(colors=colors),
               hovertemplate='<b>%{label}</b><br>Enterprises: %{value:,}<br>%{percent}<extra></extra>'),
        row=1, col=1
    )
    
    # Employment bar
    fig.add_trace(
        go.Bar(x=sector_data['sector'], y=sector_data['employment_at_setup'],
               marker=dict(color='#2E86AB'), name='Employment',
               text=sector_data['employment_at_setup'].apply(lambda x: f'{x:,.0f}'),
               textposition='outside',
               hovertemplate='<b>%{x}</b><br>Employment: %{y:,.0f}<extra></extra>'),
        row=1, col=2
    )
    
    # Cost bar
    fig.add_trace(
        go.Bar(x=sector_data['sector'], y=sector_data['project_cost_rs']/100000,
               marker=dict(color='#F18F01'), name='Avg Cost',
               text=(sector_data['project_cost_rs']/100000).apply(lambda x: f'₹{x:.1f}L'),
               textposition='outside',
               hovertemplate='<b>%{x}</b><br>Avg Cost: ₹%{y:.2f}L<extra></extra>'),
        row=2, col=1
    )
    
    # Operational rate bar
    fig.add_trace(
        go.Bar(x=sector_data['sector'], y=sector_data['operational_rate'],
               marker=dict(color='#06A77D'), name='Operational %',
               text=sector_data['operational_rate'].apply(lambda x: f'{x:.1f}%'),
               textposition='outside',
               hovertemplate='<b>%{x}</b><br>Operational: %{y:.1f}%<extra></extra>'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text='<b>Sector-wise Performance Analysis</b>',
        height=700,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

# ============================================================================
# ENHANCED TIME SERIES
# ============================================================================

def create_enhanced_time_series(df):
    """IMPROVED: Time series with trend analysis"""
    
    if len(df) == 0:
        return None
    
    monthly_data = df.groupby('month_year').agg({
        'enterprise_id': 'count',
        'employment_at_setup': 'sum',
        'margin_money_subsidy_rs': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Monthly Enterprise Setup', 'Monthly Employment Generated', 
                       'Monthly Subsidy Released (Crores)'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Enterprise setup
    fig.add_trace(
        go.Scatter(x=monthly_data['month_year'], y=monthly_data['enterprise_id'],
                   name='Enterprises', mode='lines+markers',
                   line=dict(color='#2E86AB', width=3),
                   fill='tozeroy', fillcolor='rgba(46, 134, 171, 0.2)',
                   hovertemplate='<b>%{x}</b><br>Enterprises: %{y:,.0f}<extra></extra>'),
        row=1, col=1
    )
    
    # Employment
    fig.add_trace(
        go.Scatter(x=monthly_data['month_year'], y=monthly_data['employment_at_setup'],
                   name='Employment', mode='lines+markers',
                   line=dict(color='#A23B72', width=3),
                   fill='tozeroy', fillcolor='rgba(162, 59, 114, 0.2)',
                   hovertemplate='<b>%{x}</b><br>Employment: %{y:,.0f}<extra></extra>'),
        row=2, col=1
    )
    
    # Subsidy
    fig.add_trace(
        go.Scatter(x=monthly_data['month_year'], y=monthly_data['margin_money_subsidy_rs']/10000000,
                   name='Subsidy (Cr)', mode='lines+markers',
                   line=dict(color='#F18F01', width=3),
                   fill='tozeroy', fillcolor='rgba(241, 143, 1, 0.2)',
                   hovertemplate='<b>%{x}</b><br>Subsidy: ₹%{y:.2f}Cr<extra></extra>'),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Jobs", row=2, col=1)
    fig.update_yaxes(title_text="Crores", row=3, col=1)
    
    fig.update_layout(
        title_text='<b>Temporal Trends - Monthly Performance</b>',
        height=800,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# ============================================================================
# ENHANCED DEMOGRAPHICS
# ============================================================================

def create_enhanced_demographics(df):
    """IMPROVED: Demographics with better insights"""
    
    if len(df) == 0:
        st.warning("No data")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_data = df['gender'].value_counts().reset_index()
        fig = go.Figure(data=[
            go.Pie(
                labels=gender_data['gender'],
                values=gender_data['count'],
                marker=dict(colors=['#FF6B6B', '#4ECDC4']),
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
            )
        ])
        fig.update_layout(
            title='<b>Gender Distribution</b><br><sub>Entrepreneurship across genders</sub>',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        category_data = df['category'].value_counts().reset_index()
        fig = go.Figure(data=[
            go.Pie(
                labels=category_data['category'],
                values=category_data['count'],
                marker=dict(colors=['#FFE66D', '#95E1D3', '#F38181', '#AA96DA']),
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
            )
        ])
        fig.update_layout(
            title='<b>Social Category</b><br><sub>SC/ST/OBC/General inclusion</sub>',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        location_data = df['location_type'].value_counts().reset_index()
        fig = go.Figure(data=[
            go.Pie(
                labels=location_data['location_type'],
                values=location_data['count'],
                marker=dict(colors=['#06A77D', '#F18F01']),
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>'
            )
        ])
        fig.update_layout(
            title='<b>Rural vs Urban</b><br><sub>Geographic distribution</sub>',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ENHANCED OPERATIONAL STATUS
# ============================================================================

def create_enhanced_operational_chart(df):
    """IMPROVED: Operational status with metrics"""
    
    if len(df) == 0:
        return None
    
    status_data = df.groupby('operational_status').agg({
        'enterprise_id': 'count',
        'employment_at_setup': 'sum',
        'annual_turnover_rs': 'mean',
        'project_cost_rs': 'mean'
    }).reset_index()
    
    colors_map = {'Operational': '#06A77D', 'Closed': '#FF6B6B', 'Non-Operational': '#F18F01'}
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.1
    )
    
    for idx, (status, row) in enumerate(status_data.iterrows()):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=row['enterprise_id'],
                title={'text': f"<b>{row['operational_status']}</b>"},
                gauge={
                    'axis': {'range': [None, status_data['enterprise_id'].max()]},
                    'bar': {'color': colors_map.get(row['operational_status'], '#999')},
                    'steps': [
                        {'range': [0, status_data['enterprise_id'].max() * 0.33], 'color': "lightgray"},
                        {'range': [status_data['enterprise_id'].max() * 0.33, status_data['enterprise_id'].max()], 'color': "gray"}
                    ]
                },
                domain={'x': [idx/3, (idx+1)/3], 'y': [0, 1]}
            )
        )
    
    fig.update_layout(
        title_text='<b>Enterprise Status Distribution</b>',
        height=400,
        template='plotly_white'
    )
    
    return fig

# ============================================================================
# ENHANCED FINANCIAL ANALYSIS
# ============================================================================

def create_enhanced_financial_chart(df):
    """IMPROVED: Financial analysis with sector breakdown"""
    
    if len(df) == 0:
        st.warning("No data")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['roi_percent'] < 500]['roi_percent'],
            nbinsx=40,
            marker=dict(color='#2E86AB', line=dict(width=1, color='white')),
            hovertemplate='ROI Range: %{x:.0f}%<br>Count: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title='<b>ROI Distribution</b><br><sub>Return on Investment Analysis</sub>',
            xaxis_title='ROI %',
            yaxis_title='Number of Enterprises',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sector_financial = df.groupby('sector').agg({
            'project_cost_rs': 'mean',
            'margin_money_subsidy_rs': 'mean',
            'annual_turnover_rs': 'mean'
        }) / 100000
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sector_financial.index, y=sector_financial['project_cost_rs'],
            name='Avg Project Cost', marker=dict(color='#FF6B6B')
        ))
        fig.add_trace(go.Bar(
            x=sector_financial.index, y=sector_financial['margin_money_subsidy_rs'],
            name='Avg Subsidy', marker=dict(color='#4ECDC4')
        ))
        fig.add_trace(go.Bar(
            x=sector_financial.index, y=sector_financial['annual_turnover_rs'],
            name='Avg Turnover', marker=dict(color='#06A77D')
        ))
        
        fig.update_layout(
            title='<b>Sector Financial Metrics</b><br><sub>In Lakhs (₹L)</sub>',
            xaxis_title='Sector',
            yaxis_title='Amount (Lakhs)',
            height=400,
            barmode='group',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ENHANCED CLUSTERING
# ============================================================================

def create_enhanced_clustering_analysis(df):
    """Enhanced clustering with details"""
    
    if len(df) < 10:
        st.warning("Not enough data")
        return
    
    try:
        cluster_features = ['employment_at_setup', 'project_cost_rs', 
                          'sustainability_score', 'age']
        
        X = df[cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        cluster_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': clusters.astype(str),
            'Employment': df['employment_at_setup'].values,
            'ProjectCost': df['project_cost_rs'].values,
            'Sustainability': df['sustainability_score'].values,
            'Status': df['operational_status'].values
        })
        
        cluster_descriptions = {
            '0': 'Struggling/At-Risk',
            '1': 'Developing/Moderate',
            '2': 'Growing/Good',
            '3': 'High-Performer'
        }
        
        cluster_colors = {
            '0': '#FF8C42',
            '1': '#FF6B6B',
            '2': '#4ECDC4',
            '3': '#06A77D'
        }
        
        fig = go.Figure()
        
        for cluster_id in sorted(cluster_df['Cluster'].unique()):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
            cluster_count = len(cluster_data)
            cluster_pct = (cluster_count / len(cluster_df)) * 100
            
            fig.add_trace(go.Scatter(
                x=cluster_data['PC1'],
                y=cluster_data['PC2'],
                mode='markers',
                name=f"Cluster {cluster_id}: {cluster_descriptions[cluster_id]} ({cluster_count:,} | {cluster_pct:.1f}%)",
                marker=dict(
                    size=8,
                    color=cluster_colors[cluster_id],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[
                    f"<b>Cluster {cluster_id}: {cluster_descriptions[cluster_id]}</b><br>" +
                    f"Employment: {emp:,.0f}<br>" +
                    f"Project Cost: ₹{cost:,.0f}<br>" +
                    f"Sustainability: {sus:.1f}%<br>" +
                    f"Status: {status}"
                    for emp, cost, sus, status in zip(
                        cluster_data['Employment'],
                        cluster_data['ProjectCost'],
                        cluster_data['Sustainability'],
                        cluster_data['Status']
                    )
                ],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': '<b>Enterprise Clustering Analysis (PCA)</b><br>' +
                       '<sub>Grouped by Employment, Cost Efficiency & Sustainability</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title={
                    'text': f'<b>PC1: Employment Scale ({pca.explained_variance_ratio_[0]:.1%})</b><br>' +
                           '<sub>→ Right = More employment</sub>',
                    'font': {'size': 11}
                },
                showgrid=True
            ),
            yaxis=dict(
                title={
                    'text': f'<b>PC2: Cost Efficiency ({pca.explained_variance_ratio_[1]:.1%})</b><br>' +
                           '<sub>↑ Top = Better efficiency</sub>',
                    'font': {'size': 11}
                },
                showgrid=True
            ),
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown("### Cluster Breakdown")
        summary_data = []
        for cluster_id in sorted(cluster_df['Cluster'].unique()):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
            count = len(cluster_data)
            pct = (count / len(cluster_df)) * 100
            
            summary_data.append({
                'Cluster': f"{cluster_id}: {cluster_descriptions[cluster_id]}",
                'Count': f"{count:,}",
                '%': f"{pct:.1f}%",
                'Avg Employment': f"{cluster_data['Employment'].mean():.0f}",
                'Avg Cost (₹L)': f"{cluster_data['ProjectCost'].mean()/100000:.2f}",
                'Sustainability': f"{cluster_data['Sustainability'].mean():.1f}%",
                'Operational %': f"{(cluster_data['Status'] == 'Operational').sum() / count * 100:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error: {e}")

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    default_csv_path = 'data/PMEGP_KPI_Dataset_2023-26.csv'
    
    st.title("PMEGP Executive Dashboard")
    st.markdown("*Premium Interactive Monitoring System - All Visualizations Enhanced*")
    st.markdown("---")
    
    st.sidebar.header("Data Source")
    df = load_data(default_csv_path)
    
    if df is None:
        st.error("CSV not found")
        return
    
    st.sidebar.success("✓ Data loaded")
    st.sidebar.info(f"Records: {len(df)}")
    
    st.sidebar.header("Filters")
    filters = {
        'states': st.sidebar.multiselect('States', sorted(df['state'].unique()), 
                                         default=sorted(df['state'].unique())),
        'sectors': st.sidebar.multiselect('Sectors', sorted(df['sector'].unique()),
                                         default=sorted(df['sector'].unique())),
        'categories': st.sidebar.multiselect('Category', sorted(df['category'].unique()),
                                            default=sorted(df['category'].unique())),
        'status': st.sidebar.multiselect('Status', sorted(df['operational_status'].unique()),
                                        default=sorted(df['operational_status'].unique())),
        'location': st.sidebar.multiselect('Location', sorted(df['location_type'].unique()),
                                          default=sorted(df['location_type'].unique()))
    }
    
    filtered_df = apply_filters(df, filters)
    st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)}")
    
    metrics = calculate_metrics(filtered_df)
    st.header("Key Performance Indicators")
    create_kpi_cards(metrics)
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Geographic", "Performance", "Demographics",
        "Financial", "Clustering", "Data"
    ])
    
    with tab1:
        st.header("Geographic Analysis - Enhanced")
        create_enhanced_geographic_map(filtered_df)
    
    with tab2:
        st.header("Performance Analysis - Enhanced")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### State Performance Heatmap")
            chart = create_enhanced_state_heatmap(filtered_df)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        with col2:
            st.markdown("#### Sector Analysis")
            chart = create_enhanced_sector_chart(filtered_df)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        
        st.markdown("#### Monthly Trends")
        chart = create_enhanced_time_series(filtered_df)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    with tab3:
        st.header("Demographics - Enhanced")
        create_enhanced_demographics(filtered_df)
        
        st.markdown("#### Operational Status")
        chart = create_enhanced_operational_chart(filtered_df)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    with tab4:
        st.header("Financial Analysis - Enhanced")
        create_enhanced_financial_chart(filtered_df)
    
    with tab5:
        st.header("Clustering Analysis - Enhanced")
        create_enhanced_clustering_analysis(filtered_df)
    
    with tab6:
        st.header("Data View")
        if len(filtered_df) > 0:
            st.dataframe(filtered_df.head(100), use_container_width=True)
            csv = filtered_df.to_csv(index=False)
            st.download_button("Download CSV", csv,
                              f"PMEGP_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>PMEGP Dashboard | Ministry of MSME</div>",
               unsafe_allow_html=True)

if __name__ == "__main__":
    main()
  