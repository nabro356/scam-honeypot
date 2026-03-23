"""
Disease Outbreak Dashboard
===========================
Interactive Streamlit dashboard for visualizing disease outbreak alerts
on a map of Andhra Pradesh.
Reads CSV reports from the detection scripts:
  - ip.csv + pincode_directory.csv (raw data + lat/long)
  - outbreak_report.csv (Z-score)
  - outbreak_iforest_report.csv (Isolation Forest)
  - outbreak_prophet_report.csv (Prophet)
Usage:
  pip install streamlit streamlit-folium folium plotly pandas
  streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import requests
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Outbreak Dashboard — Andhra Pradesh",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# AP center coordinates
AP_CENTER = [15.9129, 79.7400]
AP_ZOOM = 7
# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    /* KPI Card */
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.3rem 0;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    .kpi-critical .kpi-value {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-warning .kpi-value {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-success .kpi-value {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ccd6f6;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(100,255,218,0.2);
    }
    /* Severity badges */
    .badge-critical {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-alert {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #1a1a2e;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.3rem;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_ip_data():
    """Load and clean the raw IP dataset."""
    path = os.path.join(SCRIPT_DIR, "ip.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Standardize columns
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if "health" in cl and "id" in cl:
            col_map[col] = "health_id"
        elif cl in ("complaint", "complaint_code", "snomed_code", "snomed_ct_code",
                     "diagnosis", "diagnosis_code"):
            col_map[col] = "complaint"
        elif cl in ("complaint_name", "complaint_desc", "snomed_name", "ct_name",
                     "diagnosis_name", "diagnosis_desc"):
            col_map[col] = "complaint_name"
        elif cl in ("pincode", "pin_code", "zip", "postal_code"):
            col_map[col] = "pincode"
        elif cl in ("timestamp", "time_stamp", "unix_timestamp", "created_at",
                     "diagnosis_event_ts", "event_ts", "event_timestamp"):
            col_map[col] = "timestamp"
    df = df.rename(columns=col_map)
    # Parse timestamp
    sample = df["timestamp"].dropna().iloc[0] if len(df["timestamp"].dropna()) > 0 else ""
    try:
        float(sample)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    except (ValueError, TypeError):
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["datetime"].dt.date
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
    df = df.dropna(subset=["pincode", "date", "complaint_name"])
    df["pincode"] = df["pincode"].astype(int)
    df["complaint_name"] = df["complaint_name"].str.strip().str.title()
    return df
@st.cache_data
def load_pincode_directory():
    """Load India Post pincode directory for lat/long + district mapping."""
    path = os.path.join(SCRIPT_DIR, "pincode_directory.csv")
    if not os.path.exists(path):
        return None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except:
            continue
    else:
        return None
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("pincode", "pin_code", "pin"):
            col_map[col] = "pincode"
        elif cl in ("district", "district_name", "districtname"):
            col_map[col] = "district"
        elif cl in ("statename", "state_name", "state"):
            col_map[col] = "state"
        elif cl in ("latitude", "lat"):
            col_map[col] = "latitude"
        elif cl in ("longitude", "long", "lng", "lon"):
            col_map[col] = "longitude"
        elif cl in ("divisionname", "division_name", "division"):
            col_map[col] = "mandal"
        elif cl in ("taluk", "taluka", "mandal", "tehsil"):
            col_map[col] = "mandal"
    df = df.rename(columns=col_map)
    # Filter AP
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.upper().str.strip()
        df = df[df["state"].str.contains("ANDHRA", na=False)]
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df = df.dropna(subset=["pincode"])
    df["pincode"] = df["pincode"].astype(int)
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip().str.title()
    if "mandal" in df.columns:
        df["mandal"] = df["mandal"].astype(str).str.strip().str.title()
    return df
@st.cache_data
def load_alert_reports():
    """Load all outbreak report CSVs and combine them."""
    reports = {}
    files = {
        "Z-Score": "outbreak_report.csv",
        "Isolation Forest": "outbreak_iforest_report.csv",
        "Prophet": "outbreak_prophet_report.csv"
    }
    for method, fname in files.items():
        path = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df["method"] = method
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                reports[method] = df
            except Exception:
                pass
    return reports
@st.cache_data
def load_ap_geojson():
    """Load AP district GeoJSON from GitHub."""
    urls = [
        "https://raw.githubusercontent.com/geohacker/india/master/state/andhra_pradesh.geojson",
        "https://raw.githubusercontent.com/udit-001/india-maps-data/main/states/andhra-pradesh.json",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except:
            continue
    return None
def get_severity_color(severity_str):
    """Map severity string to color."""
    s = str(severity_str).lower()
    if "critical" in s:
        return "#ff4b2b"
    elif "alert" in s:
        return "#ffd200"
    return "#38ef7d"
def get_severity_icon(severity_str):
    """Map severity to folium icon."""
    s = str(severity_str).lower()
    if "critical" in s:
        return "exclamation-triangle"
    elif "alert" in s:
        return "exclamation-circle"
    return "info-sign"
# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load data ────────────────────────────────────────────────────────
    ip_df = load_ip_data()
    pin_dir = load_pincode_directory()
    reports = load_alert_reports()
    geojson = load_ap_geojson()
    if ip_df is None and not reports:
        st.error("⚠️ No data found. Please place `ip.csv` and run the detection scripts first.")
        st.info("Expected files in the same directory as this script:\n"
                "- `ip.csv`\n- `pincode_directory.csv`\n"
                "- `outbreak_report.csv` / `outbreak_iforest_report.csv` / `outbreak_prophet_report.csv`")
        return
    # ── Merge ip_df with pincode directory for lat/long ──────────────────
    merged_ip = None
    if ip_df is not None and pin_dir is not None:
        pin_geo = pin_dir[["pincode", "latitude", "longitude"]].dropna().drop_duplicates(subset=["pincode"])
        merged_ip = ip_df.merge(pin_geo, on="pincode", how="left")
        if "district" not in merged_ip.columns:
            pin_district = pin_dir[["pincode", "district"]].drop_duplicates(subset=["pincode"])
            merged_ip = merged_ip.merge(pin_district, on="pincode", how="left")
        if "mandal" in pin_dir.columns and "mandal" not in merged_ip.columns:
            pin_mandal = pin_dir[["pincode", "mandal"]].drop_duplicates(subset=["pincode"])
            merged_ip = merged_ip.merge(pin_mandal, on="pincode", how="left")
    # ── Combine all alerts ───────────────────────────────────────────────
    all_alerts = pd.concat(reports.values(), ignore_index=True) if reports else pd.DataFrame()
    # Normalize severity column
    if not all_alerts.empty:
        if "severity" not in all_alerts.columns:
            all_alerts["severity"] = "⚠️ ALERT"
        # Determine region column
        for rc in ["district", "mandal", "pincode"]:
            if rc in all_alerts.columns:
                break
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 🦠 Outbreak Dashboard")
        st.markdown("---")
        # Method filter
        available_methods = list(reports.keys()) if reports else []
        if available_methods:
            selected_methods = st.multiselect(
                "Detection Method",
                available_methods,
                default=available_methods,
                help="Which detection method's results to display"
            )
        else:
            selected_methods = []
        # Disease filter
        all_diseases = []
        if ip_df is not None:
            all_diseases = sorted(ip_df["complaint_name"].unique())
        elif not all_alerts.empty and "complaint_name" in all_alerts.columns:
            all_diseases = sorted(all_alerts["complaint_name"].unique())
        selected_diseases = st.multiselect(
            "Disease / Complaint",
            all_diseases,
            default=[],
            help="Leave empty to show all diseases"
        )
        # District filter
        all_districts = []
        if merged_ip is not None and "district" in merged_ip.columns:
            all_districts = sorted(merged_ip["district"].dropna().unique())
        selected_districts = st.multiselect(
            "District",
            all_districts,
            default=[],
            help="Leave empty to show all districts"
        )
        # Date range
        if ip_df is not None:
            min_date = pd.to_datetime(ip_df["date"].min())
            max_date = pd.to_datetime(ip_df["date"].max())
        elif not all_alerts.empty:
            min_date = pd.to_datetime(all_alerts["date"].min())
            max_date = pd.to_datetime(all_alerts["date"].max())
        else:
            min_date = pd.Timestamp("2025-01-01")
            max_date = pd.Timestamp.now()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        # Severity filter
        severity_filter = st.radio(
            "Severity",
            ["All", "🔴 CRITICAL only", "⚠️ ALERT only"],
            index=0
        )
        st.markdown("---")
        st.markdown(
            "<div style='color:#8892b0; font-size:0.75rem; text-align:center;'>"
            "Disease Outbreak Detection System<br>Andhra Pradesh</div>",
            unsafe_allow_html=True
        )
    # ── Apply filters ────────────────────────────────────────────────────
    # Filter date range
    if len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start, d_end = min_date.date(), max_date.date()
    filtered_alerts = all_alerts.copy()
    if not filtered_alerts.empty:
        if selected_methods:
            filtered_alerts = filtered_alerts[filtered_alerts["method"].isin(selected_methods)]
        if selected_diseases:
            if "complaint_name" in filtered_alerts.columns:
                filtered_alerts = filtered_alerts[filtered_alerts["complaint_name"].isin(selected_diseases)]
        if selected_districts:
            if "district" in filtered_alerts.columns:
                filtered_alerts = filtered_alerts[filtered_alerts["district"].isin(selected_districts)]
        filtered_alerts = filtered_alerts[
            (filtered_alerts["date"] >= d_start) &
            (filtered_alerts["date"] <= d_end)
        ]
        if severity_filter == "🔴 CRITICAL only":
            filtered_alerts = filtered_alerts[filtered_alerts["severity"].str.contains("CRITICAL", na=False)]
        elif severity_filter == "⚠️ ALERT only":
            filtered_alerts = filtered_alerts[~filtered_alerts["severity"].str.contains("CRITICAL", na=False)]
    filtered_ip = merged_ip
    if filtered_ip is not None:
        filtered_ip = filtered_ip[
            (filtered_ip["date"] >= d_start) &
            (filtered_ip["date"] <= d_end)
        ]
        if selected_diseases:
            filtered_ip = filtered_ip[filtered_ip["complaint_name"].isin(selected_diseases)]
        if selected_districts and "district" in filtered_ip.columns:
            filtered_ip = filtered_ip[filtered_ip["district"].isin(selected_districts)]
    # ── Title ────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center; font-size:1.8rem; font-weight:700; "
        "background: linear-gradient(135deg, #00d2ff, #3a7bd5); "
        "-webkit-background-clip: text; -webkit-text-fill-color: transparent;'>"
        "🦠 Disease Outbreak Detection — Andhra Pradesh</h1>",
        unsafe_allow_html=True
    )
    # ── KPI Cards ────────────────────────────────────────────────────────
    total_alerts = len(filtered_alerts) if not filtered_alerts.empty else 0
    critical_count = (
        filtered_alerts["severity"].str.contains("CRITICAL", na=False).sum()
        if not filtered_alerts.empty else 0
    )
    districts_affected = (
        filtered_alerts["district"].nunique()
        if not filtered_alerts.empty and "district" in filtered_alerts.columns else 0
    )
    top_disease = (
        filtered_alerts["complaint_name"].value_counts().index[0]
        if not filtered_alerts.empty and "complaint_name" in filtered_alerts.columns
        and len(filtered_alerts["complaint_name"].value_counts()) > 0
        else "—"
    )
    total_cases = len(filtered_ip) if filtered_ip is not None else 0
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Cases</div>'
            f'<div class="kpi-value">{total_cases:,}</div></div>',
            unsafe_allow_html=True
        )
    with k2:
        css_class = "kpi-warning" if total_alerts > 0 else ""
        st.markdown(
            f'<div class="kpi-card {css_class}"><div class="kpi-label">Total Alerts</div>'
            f'<div class="kpi-value">{total_alerts}</div></div>',
            unsafe_allow_html=True
        )
    with k3:
        css_class = "kpi-critical" if critical_count > 0 else ""
        st.markdown(
            f'<div class="kpi-card {css_class}"><div class="kpi-label">Critical</div>'
            f'<div class="kpi-value">{critical_count}</div></div>',
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card kpi-success"><div class="kpi-label">Districts Affected</div>'
            f'<div class="kpi-value">{districts_affected}</div></div>',
            unsafe_allow_html=True
        )
    with k5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Top Disease</div>'
            f'<div class="kpi-value" style="font-size:1rem;">{top_disease}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)
    # ── Map + Disease Chart (side by side) ───────────────────────────────
    map_col, chart_col = st.columns([3, 2])
    with map_col:
        st.markdown('<div class="section-header">🗺️ Outbreak Hotspot Map</div>',
                    unsafe_allow_html=True)
        # Create Folium map
        m = folium.Map(
            location=AP_CENTER,
            zoom_start=AP_ZOOM,
            tiles="CartoDB dark_matter",
            control_scale=True
        )
        # Add GeoJSON district boundaries
        if geojson:
            folium.GeoJson(
                geojson,
                name="District Boundaries",
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": "#3a7bd5",
                    "weight": 1.5,
                    "fillOpacity": 0.05,
                    "dashArray": "3"
                }
            ).add_to(m)
        # ── Build hotspot data from ip + pincode directory ──────────────
        if filtered_ip is not None and "latitude" in filtered_ip.columns:
            # Aggregate by pincode for map markers
            pin_agg = (
                filtered_ip
                .dropna(subset=["latitude", "longitude"])
                .groupby(["pincode", "latitude", "longitude"])
                .agg(
                    total_cases=("health_id", "count"),
                    diseases=("complaint_name", lambda x: ", ".join(x.unique()[:3])),
                    disease_count=("complaint_name", "nunique")
                )
                .reset_index()
            )
            # Get district for each pincode
            if "district" in filtered_ip.columns:
                pin_district = (
                    filtered_ip[["pincode", "district"]]
                    .drop_duplicates(subset=["pincode"])
                )
                pin_agg = pin_agg.merge(pin_district, on="pincode", how="left")
            # Add heatmap layer
            if len(pin_agg) > 0:
                heat_data = pin_agg[["latitude", "longitude", "total_cases"]].values.tolist()
                HeatMap(
                    heat_data,
                    name="Case Heatmap",
                    radius=20,
                    blur=15,
                    max_zoom=13,
                    gradient={0.2: '#38ef7d', 0.5: '#ffd200', 0.8: '#ff6b35', 1: '#ff4b2b'}
                ).add_to(m)
            # Add alert markers if alerts have pincode matching
            if not filtered_alerts.empty:
                alert_group = folium.FeatureGroup(name="Alert Markers")
                # Try to match alerts to lat/long via pincode
                alert_with_geo = filtered_alerts.copy()
                if "pincode" in alert_with_geo.columns and pin_dir is not None:
                    pin_geo = pin_dir[["pincode", "latitude", "longitude"]].dropna().drop_duplicates(subset=["pincode"])
                    alert_with_geo["pincode"] = pd.to_numeric(alert_with_geo["pincode"], errors="coerce")
                    alert_with_geo = alert_with_geo.merge(pin_geo, on="pincode", how="left")
                elif "district" in alert_with_geo.columns and pin_dir is not None:
                    # Use district centroid
                    district_centroids = (
                        pin_dir.dropna(subset=["latitude", "longitude"])
                        .groupby("district")
                        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
                        .reset_index()
                    )
                    if "latitude" not in alert_with_geo.columns:
                        alert_with_geo = alert_with_geo.merge(district_centroids, on="district", how="left")
                if "latitude" in alert_with_geo.columns:
                    alert_with_geo = alert_with_geo.dropna(subset=["latitude", "longitude"])
                    for _, row in alert_with_geo.iterrows():
                        severity = str(row.get("severity", "ALERT"))
                        color = get_severity_color(severity)
                        disease = row.get("complaint_name", "Unknown")
                        case_count = row.get("case_count", row.get("actual", "N/A"))
                        method = row.get("method", "")
                        date = row.get("date", "")
                        popup_html = f"""
                        <div style="font-family: Inter, sans-serif; min-width: 180px;">
                            <b style="color:{color}; font-size:14px;">{severity}</b><br>
                            <b>{disease}</b><br>
                            📅 {date}<br>
                            📊 Cases: {case_count}<br>
                            🔬 Method: {method}
                        </div>
                        """
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=8,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            popup=folium.Popup(popup_html, max_width=250),
                        ).add_to(alert_group)
                alert_group.add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=480, use_container_width=True)
    with chart_col:
        st.markdown('<div class="section-header">📊 Disease Distribution</div>',
                    unsafe_allow_html=True)
        if not filtered_alerts.empty and "complaint_name" in filtered_alerts.columns:
            disease_counts = (
                filtered_alerts["complaint_name"]
                .value_counts()
                .head(15)
                .reset_index()
            )
            disease_counts.columns = ["Disease", "Alerts"]
            fig_bar = px.bar(
                disease_counts,
                x="Alerts",
                y="Disease",
                orientation="h",
                color="Alerts",
                color_continuous_scale=["#38ef7d", "#ffd200", "#ff4b2b"],
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccd6f6", family="Inter"),
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=0),
                yaxis=dict(autorange="reversed"),
                height=220,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No alert data to display.")
        # District breakdown
        st.markdown('<div class="section-header">🏛️ Alerts by District</div>',
                    unsafe_allow_html=True)
        if not filtered_alerts.empty and "district" in filtered_alerts.columns:
            dist_counts = (
                filtered_alerts
                .groupby("district")
                .agg(
                    alerts=("severity", "count"),
                    critical=("severity", lambda x: x.str.contains("CRITICAL", na=False).sum())
                )
                .reset_index()
                .sort_values("alerts", ascending=False)
                .head(10)
            )
            fig_dist = px.bar(
                dist_counts,
                x="district",
                y=["alerts", "critical"],
                barmode="overlay",
                color_discrete_map={"alerts": "#3a7bd5", "critical": "#ff4b2b"},
                labels={"value": "Count", "district": "District", "variable": "Type"}
            )
            fig_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccd6f6", family="Inter"),
                margin=dict(l=0, r=10, t=10, b=0),
                height=200,
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No district data available.")
    # ── Time Series Chart ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Daily Case Trend</div>',
                unsafe_allow_html=True)
    if filtered_ip is not None and len(filtered_ip) > 0:
        # Let user pick a disease for the time series
        ts_diseases = sorted(filtered_ip["complaint_name"].unique())
        ts_cols = st.columns([3, 1])
        with ts_cols[0]:
            ts_disease = st.selectbox(
                "Select disease for time series",
                ts_diseases,
                index=0 if ts_diseases else None,
                label_visibility="collapsed"
            )
        if ts_disease:
            daily = (
                filtered_ip[filtered_ip["complaint_name"] == ts_disease]
                .groupby("date")
                .agg(cases=("health_id", "count"))
                .reset_index()
                .sort_values("date")
            )
            fig_ts = go.Figure()
            # Case count line
            fig_ts.add_trace(go.Scatter(
                x=daily["date"], y=daily["cases"],
                mode="lines",
                name="Daily Cases",
                line=dict(color="#3a7bd5", width=2),
                fill="tozeroy",
                fillcolor="rgba(58,123,213,0.1)"
            ))
            # 7-day moving average
            daily["ma7"] = daily["cases"].rolling(7, min_periods=1).mean()
            fig_ts.add_trace(go.Scatter(
                x=daily["date"], y=daily["ma7"],
                mode="lines",
                name="7-day Average",
                line=dict(color="#00d2ff", width=2, dash="dash")
            ))
            # Overlay alert markers
            if not filtered_alerts.empty and "complaint_name" in filtered_alerts.columns:
                disease_alerts = filtered_alerts[
                    filtered_alerts["complaint_name"].str.lower() == ts_disease.lower()
                ]
                if not disease_alerts.empty:
                    alert_dates = disease_alerts["date"].unique()
                    alert_cases = daily[daily["date"].isin(alert_dates)]
                    if len(alert_cases) > 0:
                        fig_ts.add_trace(go.Scatter(
                            x=alert_cases["date"], y=alert_cases["cases"],
                            mode="markers",
                            name="⚠️ Alert",
                            marker=dict(color="#ff4b2b", size=10, symbol="triangle-up",
                                       line=dict(width=1, color="white"))
                        ))
            fig_ts.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccd6f6", family="Inter"),
                margin=dict(l=0, r=10, t=30, b=0),
                height=300,
                legend=dict(orientation="h", y=1.1),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Cases"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No case data available for time series.")
    # ── Alert Table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Alert Details</div>',
                unsafe_allow_html=True)
    if not filtered_alerts.empty:
        # Select display columns
        display_cols = []
        for col in ["severity", "method", "level", "district", "mandal", "pincode",
                     "complaint_name", "date", "case_count", "actual",
                     "rolling_mean", "z_score", "anomaly_score",
                     "yhat", "yhat_upper", "excess"]:
            if col in filtered_alerts.columns:
                display_cols.append(col)
        display_df = filtered_alerts[display_cols].copy()
        # Round numeric columns
        for col in ["rolling_mean", "z_score", "anomaly_score", "yhat", "yhat_upper", "excess"]:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
        display_df = display_df.sort_values(
            "date", ascending=False
        ).reset_index(drop=True)
        # Method tabs
        if len(selected_methods) > 1:
            tabs = st.tabs(["All"] + selected_methods)
            with tabs[0]:
                st.dataframe(display_df, use_container_width=True, height=350)
            for i, method in enumerate(selected_methods):
                with tabs[i + 1]:
                    method_df = display_df[display_df["method"] == method] if "method" in display_df.columns else display_df
                    st.dataframe(method_df, use_container_width=True, height=350)
        else:
            st.dataframe(display_df, use_container_width=True, height=350)
        # Download button
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Alerts CSV",
            csv_data,
            "filtered_alerts.csv",
            "text/csv"
        )
    else:
        st.info("No alerts match the current filters. Try adjusting the sidebar filters.")
if __name__ == "__main__":
    main()
