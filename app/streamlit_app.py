"""
AIOps Infrastructure Log Anomaly Detection — Streamlit Triage Dashboard
Designed for proactive enterprise infrastructure monitoring and SRE incident investigation.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page settings
st.set_page_config(
    page_title="AIOps Infrastructure Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphism & Modern Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 18px 22px;
        border-radius: 12px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.5);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.88rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700;
        font-size: 1.95rem;
    }
    
    /* Highlighted table header */
    th {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    
    /* Status pills */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .status-live { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .status-alert { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }
</style>
""", unsafe_allow_html=True)


# Helper function to find paths regardless of launch dir (app/ or root)
@st.cache_data
def load_data():
    base_dir = Path(__file__).resolve().parent.parent
    
    # Paths
    preds_path = base_dir / "data" / "processed" / "all_predictions.csv"
    feat_path = base_dir / "data" / "processed" / "features.csv"
    overall_path = base_dir / "results" / "overall_comparison.csv"
    type_path = base_dir / "results" / "per_type_metrics.csv"
    top_path = base_dir / "results" / "top_anomalies.csv"
    
    if not preds_path.exists():
        st.error(f"Missing master predictions file: {preds_path}. Please run Notebook 05/06 first.")
        st.stop()
        
    df_preds = pd.read_csv(preds_path, parse_dates=['timestamp'])
    
    df_feats = None
    if feat_path.exists():
        df_feats = pd.read_csv(feat_path, parse_dates=['timestamp'])
        # Merge raw key metrics into predictions dataframe for rich plotting
        raw_cols = ['cpu_utilization_pct', 'memory_utilization_pct', 'response_time_ms', 'error_rate_pct', 'requests_per_min']
        avail_cols = [c for c in raw_cols if c in df_feats.columns]
        if avail_cols:
            df_preds = pd.merge(df_preds, df_feats[['timestamp', 'server_id'] + avail_cols], on=['timestamp', 'server_id'], how='left')
            
    df_overall = pd.read_csv(overall_path) if overall_path.exists() else None
    df_types = pd.read_csv(type_path) if type_path.exists() else None
    df_top = pd.read_csv(top_path) if top_path.exists() else None
    
    return df_preds, df_overall, df_types, df_top, base_dir


df, df_overall, df_types, df_top, base_dir = load_data()

# Sidebar Controls & Navigation
st.sidebar.markdown("### AIOps Infrastructure Monitor")
st.sidebar.markdown('<span class="status-pill status-live">● Live Monitoring Active</span>', unsafe_allow_html=True)
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation",
    options=[
        "Fleet Health & KPI Center",
        "SRE Incident Triage Queue",
        "Multi-Model Diagnostic Timeline",
        "Model Architecture & Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Global Fleet Filters")

# Server filter
all_servers = ["All Servers"] + sorted(df['server_id'].unique().tolist())
selected_server = st.sidebar.selectbox("Select Server ID:", options=all_servers)

# Service filter
all_services = ["All Services"] + sorted(df['service_type'].unique().tolist())
selected_service = st.sidebar.selectbox("Select Service Type:", options=all_services)

# Filter dataset based on selections
filtered_df = df.copy()
if selected_server != "All Servers":
    filtered_df = filtered_df[filtered_df['server_id'] == selected_server]
if selected_service != "All Services":
    filtered_df = filtered_df[filtered_df['service_type'] == selected_service]


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: FLEET HEALTH & KPI CENTER
# ─────────────────────────────────────────────────────────────────────────────
if nav_choice == "Fleet Health & KPI Center":
    st.title("Enterprise Fleet Health & KPI Center")
    st.markdown("Proactive, unsupervised anomaly detection running across **15 enterprise servers** over a **30-day time horizon**.")
    
    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Logs Analyzed", f"{len(filtered_df):,}")
    with col2:
        km_flags = int(filtered_df['kmeans_anomaly'].sum()) if 'kmeans_anomaly' in filtered_df else 0
        st.metric("K-Means (PCA) Alerts", f"{km_flags:,}", delta=f"{(km_flags/max(1,len(filtered_df)))*100:.1f}% rate", delta_color="inverse")
    with col3:
        db_flags = int(filtered_df['dbscan_anomaly'].sum()) if 'dbscan_anomaly' in filtered_df else 0
        st.metric("DBSCAN (PCA) Alerts", f"{db_flags:,}", delta=f"{(db_flags/max(1,len(filtered_df)))*100:.1f}% rate", delta_color="inverse")
    with col4:
        if_flags = int(filtered_df['iforest_anomaly'].sum()) if 'iforest_anomaly' in filtered_df else 0
        st.metric("Isolation Forest Alerts", f"{if_flags:,}", delta=f"{(if_flags/max(1,len(filtered_df)))*100:.1f}% rate", delta_color="inverse")

    st.markdown("---")
    
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.subheader("Server Fleet Anomaly Breakdown")
        if not filtered_df.empty:
            server_agg = filtered_df.groupby(['server_id', 'service_type']).agg(
                Total_Logs=('timestamp', 'count'),
                KMeans_Alerts=('kmeans_anomaly', 'sum'),
                DBSCAN_Alerts=('dbscan_anomaly', 'sum'),
                IForest_Alerts=('iforest_anomaly', 'sum'),
            ).reset_index()
            server_agg['Max_Alert_Rate_%'] = round((server_agg[['KMeans_Alerts', 'DBSCAN_Alerts', 'IForest_Alerts']].max(axis=1) / server_agg['Total_Logs']) * 100, 1)
            server_agg = server_agg.sort_values('Max_Alert_Rate_%', ascending=False)
            st.dataframe(server_agg, use_container_width=True, hide_index=True)
        else:
            st.warning("No data matches current filter criteria.")
            
    with col_right:
        st.subheader("Fleet-Wide Alert Distribution by Detector")
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
            ax.set_facecolor('none')
            alert_counts = [
                filtered_df['kmeans_anomaly'].sum(),
                filtered_df['dbscan_anomaly'].sum(),
                filtered_df['iforest_anomaly'].sum()
            ]
            colors = ['#38bdf8', '#34d399', '#a855f7']
            bars = ax.bar(['K-Means (PCA)', 'DBSCAN (PCA)', 'Isolation Forest'], alert_counts, color=colors, width=0.55)
            ax.set_ylabel('Total Alerts Flagged', color='#cbd5e1', fontweight='bold')
            ax.tick_params(colors='#cbd5e1')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height):,}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),  # 4 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color='#f8fafc', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SRE INCIDENT TRIAGE QUEUE
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "SRE Incident Triage Queue":
    st.title("SRE 1-Hour Rolling Incident Triage Queue")
    st.markdown("To prevent alert fatigue from noisy 1-minute point-in-time flags, detections are aggregated into **1-hour rolling server windows**. Severity is ranked by **Composite Incident Score** (`sum of flags across all three models inside the hour`).")
    
    if df_top is not None:
        st.markdown("#### Top Most Severe Incident Windows Across Fleet")
        
        # Display nicely styled triage dataframe
        display_top = df_top.copy()
        if 'hour_window' in display_top:
            display_top['hour_window'] = pd.to_datetime(display_top['hour_window']).dt.strftime('%Y-%m-%d %H:%00')
            
        cols_to_show = ['hour_window', 'server_id', 'service_type', 'composite_incident_score', 'kmeans_flags', 'iforest_flags', 'dbscan_flags']
        if 'dominant_true_type' in display_top.columns:
            cols_to_show.append('dominant_true_type')
            
        st.dataframe(
            display_top[cols_to_show],
            use_container_width=True,
            hide_index=True,
            column_config={
                "hour_window": "Incident Hour Window",
                "server_id": "Server ID",
                "service_type": "Service Type",
                "composite_incident_score": st.column_config.ProgressColumn(
                    "Composite Severity Score",
                    help="Aggregate intensity of alerts across all 3 unsupervised models in this 1-hour window",
                    format="%d",
                    min_value=0,
                    max_value=int(display_top['composite_incident_score'].max() or 12),
                ),
                "kmeans_flags": "K-Means Flags",
                "iforest_flags": "IForest Flags",
                "dbscan_flags": "DBSCAN Flags",
                "dominant_true_type": "True Signature (Evaluation Only)"
            }
        )
        
        st.markdown("---")
        st.info("**Triage Advice for SREs:** Click any row above or use the Server dropdown in the left sidebar to jump into the **Multi-Model Diagnostic Timeline** and inspect the raw metric anomalies during these exact hours.")
    else:
        st.warning("Top anomalies summary (`results/top_anomalies.csv`) not found. Please run Notebook 06.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MULTI-MODEL DIAGNOSTIC TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "Multi-Model Diagnostic Timeline":
    st.title("Multi-Model Diagnostic Timeline & Metric Inspection")
    
    if selected_server == "All Servers":
        st.warning("Please select a specific **Server ID** (e.g., `srv-web-01` or `srv-api-02`) from the left sidebar to view detailed time-series metrics and model alert markers.")
    else:
        server_data = filtered_df.sort_values('timestamp').copy()
        st.markdown(f"Inspecting telemetry for **`{selected_server}`** (`{server_data['service_type'].iloc[0] if not server_data.empty else 'N/A'}`) — `{len(server_data):,}` data points.")
        
        # Metric Selector
        avail_metrics = ['cpu_utilization_pct', 'memory_utilization_pct', 'response_time_ms', 'error_rate_pct', 'requests_per_min']
        avail_metrics = [m for m in avail_metrics if m in server_data.columns]
        
        if avail_metrics:
            chosen_metric = st.selectbox("Select Infrastructure Metric to Visualize:", options=avail_metrics, format_func=lambda x: x.replace('_', ' ').title())
            
            # Interactive Streamlit Line/Area Chart for raw metric
            chart_data = server_data.set_index('timestamp')[[chosen_metric]]
            st.markdown(f"#### `{chosen_metric.replace('_', ' ').title()}` Time-Series Trend")
            st.line_chart(chart_data, color="#38bdf8")
            
            st.markdown("#### Multi-Model Alert Overlay Timeline")
            # Create a clean multi-panel timeline using matplotlib
            fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True, facecolor='none')
            for ax in axes:
                ax.set_facecolor('#0f172a')
                ax.tick_params(colors='#cbd5e1')
                for spine in ax.spines.values():
                    spine.set_color('#334155')
                    
            # 1. Chosen Raw Metric
            axes[0].plot(server_data['timestamp'], server_data[chosen_metric], color='#38bdf8', lw=1.2)
            axes[0].set_ylabel(chosen_metric.split('_')[0].upper(), color='#38bdf8', fontweight='bold', fontsize=9)
            axes[0].set_title(f"{selected_server} -- Telemetry vs Detector Flags", color='#f8fafc', fontweight='bold', fontsize=11)
            
            # 2. K-Means
            axes[1].plot(server_data['timestamp'], server_data['kmeans_anomaly'], color='#38bdf8', lw=1.2, drawstyle='steps-post')
            axes[1].fill_between(server_data['timestamp'], server_data['kmeans_anomaly'], color='#38bdf8', alpha=0.3)
            axes[1].set_ylabel('K-Means\nFlag', color='#38bdf8', fontweight='bold', fontsize=9); axes[1].set_yticks([0, 1])
            
            # 3. DBSCAN
            axes[2].plot(server_data['timestamp'], server_data['dbscan_anomaly'], color='#34d399', lw=1.2, drawstyle='steps-post')
            axes[2].fill_between(server_data['timestamp'], server_data['dbscan_anomaly'], color='#34d399', alpha=0.3)
            axes[2].set_ylabel('DBSCAN\nFlag', color='#34d399', fontweight='bold', fontsize=9); axes[2].set_yticks([0, 1])
            
            # 4. Isolation Forest
            axes[3].plot(server_data['timestamp'], server_data['iforest_anomaly'], color='#a855f7', lw=1.2, drawstyle='steps-post')
            axes[3].fill_between(server_data['timestamp'], server_data['iforest_anomaly'], color='#a855f7', alpha=0.3)
            axes[3].set_ylabel('IForest\nFlag', color='#a855f7', fontweight='bold', fontsize=9); axes[3].set_yticks([0, 1])
            axes[3].set_xlabel('Timestamp', color='#cbd5e1', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Raw metrics (cpu_utilization_pct, etc.) not found inside predictions table. Make sure features.csv is available.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: MODEL ARCHITECTURE & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
elif nav_choice == "Model Architecture & Evaluation":
    st.title("Unsupervised Model Architecture & Objective Evaluation")
    st.markdown("Ground-truth labels (`is_anomaly`, `anomaly_type`) were unsealed **only after** all three unsupervised models completed predictions. This section documents objective software engineering metrics and architectural tradeoffs.")
    
    if df_overall is not None:
        st.markdown("### Overall Detection Performance Against Ground Truth (`is_anomaly`)")
        st.dataframe(df_overall, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        if df_types is not None:
            st.markdown("### Sensitivity Across Anomaly Signatures (`per_type_metrics.csv`)")
            st.dataframe(df_types, use_container_width=True, hide_index=True)
            
            st.markdown("""
            #### Engineering Insights & Tradeoffs:
            * **Sudden Spikes (`error_cascade`: 100% recall, `latency_spike`: 90–98% recall):** Point-in-time density and distance thresholds isolate sharp, correlated metric deviations effortlessly.
            * **Gradual Drift (`memory_leak`: 2.9–10.7% recall):** Memory leaks climb slowly inside normal baseline bounds over 6–12 hours. Stateless point-in-time models struggle to catch them early—demonstrating why multi-hour rolling derivative features (`_roll_mean_4h`, `_diff`) and composite 1-hour window aggregation (`composite_incident_score`) are essential for production monitoring.
            * **PCA vs. Raw Feature Space:** Isolation Forest operating natively in 42D space caught single-axis deviations well (`cpu_spike`), while PCA + K-Means achieved superior consistency on multi-metric correlated faults.
            """)
    with col_b:
        st.markdown("### Diagnostic Visualizations")
        # Check and display diagnostic images
        img_recall = base_dir / "results" / "per_type_recall.png"
        img_iforest = base_dir / "results" / "iforest_scatter.png"
        
        if img_recall.exists():
            st.image(str(img_recall), caption="Per-Anomaly-Type Recall Comparison Bar Chart", use_column_width=True)
        if img_iforest.exists():
            st.image(str(img_iforest), caption="Isolation Forest 2D Projections (PCA Space vs Raw Space)", use_column_width=True)

st.markdown("---")
st.caption("**AIOps Infrastructure Anomaly Detection** — Built with Python, Scikit-Learn, and Streamlit.")
