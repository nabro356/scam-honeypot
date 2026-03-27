"""
Disease Outbreak Detection — Isolation Forest with Adaptive Contamination + Spatial Features
==============================================================================================
The most advanced variant. Adds spatial (geographic spread) features on top of
adaptive contamination.
Key improvements over base:
  1. Adaptive contamination (dynamic per disease-region pair)
  2. Enhanced time-series features (9 features)
  3. Spatial features using lat/long from pincode_directory.csv:
     - neighbor_cases:    total cases in nearby pincodes (within RADIUS_KM)
     - spatial_ratio:     this pincode's cases / neighbor average
     - neighbor_alert:    are neighbors also spiking?
     These catch outbreaks that are SPREADING geographically.
Usage:
  # Standalone:
  python outbreak_iforest_spatial.py
  # From notebook with Impala DataFrame:
  from outbreak_iforest_spatial import main
  main(ip_df=your_dataframe)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
import os
import sys
warnings.filterwarnings("ignore")
from data_utils import (
    load_ip_dataset, load_pincode_mapping, merge_geography,
    build_daily_series, print_header, print_subheader, SCRIPT_DIR
)
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Adaptive contamination
MIN_CONTAMINATION = 0.01
MAX_CONTAMINATION = 0.10
BASE_CASES = 50
# Spatial parameters
RADIUS_KM = 50               # Neighbor radius in km
                              # Pincodes within 50km are considered "neighbors"
# Isolation Forest
N_ESTIMATORS = 200
RANDOM_STATE = 42
# Feature engineering
ROLLING_WINDOW = 7
MIN_TOTAL_CASES = 20
MIN_DAYS = 14
# Severity
CRITICAL_SCORE_THRESHOLD = -0.3
CRITICAL_SPIKE_RATIO = 5.0
# Output
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "outbreak_iforest_spatial_report.csv")
# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))
def build_neighbor_map(pincode_coords, radius_km=RADIUS_KM):
    """
    Build a dict: pincode → list of neighbor pincodes within radius_km.
    
    Uses Haversine distance on lat/long from pincode_directory.csv.
    """
    pincodes = pincode_coords["pincode"].values
    lats = pincode_coords["latitude"].values
    lons = pincode_coords["longitude"].values
    
    n = len(pincodes)
    print(f"  Building neighbor map for {n} pincodes (radius={radius_km}km)...")
    
    # Compute pairwise distances using vectorized Haversine
    coords = np.column_stack([lats, lons])
    
    # For efficiency, process in chunks if too many pincodes
    neighbors = {}
    
    if n <= 2000:
        # Compute full distance matrix
        coords_rad = np.radians(coords)
        
        for i in range(n):
            distances = np.array([
                haversine_km(lats[i], lons[i], lats[j], lons[j])
                for j in range(n)
            ])
            mask = (distances <= radius_km) & (distances > 0)  # Exclude self
            neighbors[pincodes[i]] = pincodes[mask].tolist()
    else:
        # For large datasets, use a simpler bounding-box pre-filter
        # ~1 degree latitude ≈ 111 km
        deg_threshold = radius_km / 111.0 * 1.5  # 1.5x for safety margin
        
        for i in range(n):
            lat_mask = np.abs(lats - lats[i]) <= deg_threshold
            lon_mask = np.abs(lons - lons[i]) <= deg_threshold
            candidates = np.where(lat_mask & lon_mask)[0]
            
            nearby = []
            for j in candidates:
                if j != i:
                    d = haversine_km(lats[i], lons[i], lats[j], lons[j])
                    if d <= radius_km:
                        nearby.append(pincodes[j])
            neighbors[pincodes[i]] = nearby
    
    avg_neighbors = np.mean([len(v) for v in neighbors.values()])
    print(f"  Average neighbors per pincode: {avg_neighbors:.1f}")
    
    return neighbors
def compute_contamination(total_cases):
    """Adaptive contamination based on volume."""
    raw = BASE_CASES / max(total_cases, 1)
    return max(MIN_CONTAMINATION, min(MAX_CONTAMINATION, raw))
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (with spatial)
# ─────────────────────────────────────────────────────────────────────────────
def build_features_with_spatial(daily_df, neighbor_map, daily_pincode_cases):
    """
    Build features including spatial neighbor features.
    
    12 features per day:
      Time-series (9):
        1. case_count, 2. rolling_7d_mean, 3. rolling_7d_std,
        4. rolling_14d_mean, 5. ratio_to_mean, 6. diff_1d,
        7. day_of_week, 8. lag_7d_count, 9. cumulative_3d
      Spatial (3):
        10. neighbor_cases:     total cases in nearby pincodes today
        11. spatial_ratio:      this pincode's cases / neighbor avg cases
        12. neighbor_spike:     max spike ratio among neighbors
    """
    all_features = []
    pair_stats = []
    
    for (pincode, disease), group in daily_df.groupby(["pincode", "complaint_name"]):
        group = group.sort_values("date").copy()
        
        total_cases = group["case_count"].sum()
        if total_cases < MIN_TOTAL_CASES or len(group) < MIN_DAYS:
            continue
        
        # ── Time-series features ───────────────────────────────────────
        group["rolling_7d_mean"] = (
            group["case_count"].rolling(ROLLING_WINDOW, min_periods=3).mean().shift(1)
        )
        group["rolling_7d_std"] = (
            group["case_count"].rolling(ROLLING_WINDOW, min_periods=3).std().shift(1)
        )
        group["rolling_14d_mean"] = (
            group["case_count"].rolling(14, min_periods=5).mean().shift(1)
        )
        group["ratio_to_mean"] = np.where(
            group["rolling_7d_mean"] > 0,
            group["case_count"] / group["rolling_7d_mean"],
            group["case_count"]
        )
        group["diff_1d"] = group["case_count"].diff()
        group["day_of_week"] = pd.to_datetime(group["date"]).dt.dayofweek
        group["lag_7d_count"] = group["case_count"].shift(7)
        group["cumulative_3d"] = group["case_count"].rolling(3, min_periods=1).sum()
        
        # ── Spatial features ───────────────────────────────────────────
        neighbor_pincodes = neighbor_map.get(pincode, [])
        
        neighbor_cases_list = []
        spatial_ratio_list = []
        neighbor_spike_list = []
        
        for _, row in group.iterrows():
            date = row["date"]
            my_cases = row["case_count"]
            
            # Get neighbor cases for this disease on this date
            n_cases = 0
            n_count = 0
            max_neighbor_ratio = 0
            
            for np_code in neighbor_pincodes:
                key = (np_code, disease, date)
                if key in daily_pincode_cases:
                    nc = daily_pincode_cases[key]
                    n_cases += nc
                    n_count += 1
                    
                    # Check if neighbor is also spiking
                    # (compare to their 7d mean approximation)
                    prev_key = (np_code, disease, date - pd.Timedelta(days=1))
                    if prev_key in daily_pincode_cases:
                        prev = daily_pincode_cases[prev_key]
                        if prev > 0:
                            max_neighbor_ratio = max(max_neighbor_ratio, nc / prev)
            
            neighbor_avg = n_cases / max(n_count, 1)
            spatial_ratio = my_cases / max(neighbor_avg, 0.1) if n_count > 0 else 1.0
            
            neighbor_cases_list.append(n_cases)
            spatial_ratio_list.append(spatial_ratio)
            neighbor_spike_list.append(max_neighbor_ratio)
        
        group["neighbor_cases"] = neighbor_cases_list
        group["spatial_ratio"] = spatial_ratio_list
        group["neighbor_spike"] = neighbor_spike_list
        
        # Drop rows where features aren't ready
        group = group.dropna(subset=[
            "rolling_7d_mean", "rolling_7d_std", "diff_1d",
            "rolling_14d_mean", "lag_7d_count"
        ])
        
        if len(group) < MIN_DAYS:
            continue
        
        contam = compute_contamination(total_cases)
        group["_contamination"] = contam
        group["_total_cases"] = total_cases
        
        all_features.append(group)
        pair_stats.append({
            "pincode": pincode, "disease": disease,
            "total_cases": total_cases, "neighbors": len(neighbor_pincodes),
            "contamination": contam
        })
    
    if pair_stats:
        stats_df = pd.DataFrame(pair_stats)
        print(f"  Pairs to analyze: {len(stats_df)}")
        print(f"  Contamination range: {stats_df['contamination'].min():.3f} — "
              f"{stats_df['contamination'].max():.3f}")
        print(f"  Avg neighbors: {stats_df['neighbors'].mean():.1f}")
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
# Non-spatial version for mandal/district levels (no lat/long per mandal)
def build_features_standard(daily_df, region_col):
    """Build features for mandal/district levels (no spatial)."""
    all_features = []
    
    for (region, disease), group in daily_df.groupby([region_col, "complaint_name"]):
        group = group.sort_values("date").copy()
        
        total_cases = group["case_count"].sum()
        if total_cases < MIN_TOTAL_CASES or len(group) < MIN_DAYS:
            continue
        
        group["rolling_7d_mean"] = group["case_count"].rolling(ROLLING_WINDOW, min_periods=3).mean().shift(1)
        group["rolling_7d_std"] = group["case_count"].rolling(ROLLING_WINDOW, min_periods=3).std().shift(1)
        group["rolling_14d_mean"] = group["case_count"].rolling(14, min_periods=5).mean().shift(1)
        group["ratio_to_mean"] = np.where(
            group["rolling_7d_mean"] > 0,
            group["case_count"] / group["rolling_7d_mean"],
            group["case_count"]
        )
        group["diff_1d"] = group["case_count"].diff()
        group["day_of_week"] = pd.to_datetime(group["date"]).dt.dayofweek
        group["lag_7d_count"] = group["case_count"].shift(7)
        group["cumulative_3d"] = group["case_count"].rolling(3, min_periods=1).sum()
        
        group = group.dropna(subset=["rolling_7d_mean", "rolling_7d_std", "diff_1d",
                                      "rolling_14d_mean", "lag_7d_count"])
        if len(group) < MIN_DAYS:
            continue
        
        contam = compute_contamination(total_cases)
        group["_contamination"] = contam
        group["_total_cases"] = total_cases
        
        all_features.append(group)
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies(features_df, region_col, feature_cols):
    """Run Isolation Forest with adaptive contamination."""
    all_results = []
    anomaly_count = 0
    total_pairs = 0
    
    for (region, disease), group in features_df.groupby([region_col, "complaint_name"]):
        total_pairs += 1
        group = group.copy()
        
        contam = group["_contamination"].iloc[0]
        X = group[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contam,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        group["anomaly_label"] = model.fit_predict(X_scaled)
        group["anomaly_score"] = model.decision_function(X_scaled)
        
        anomalies = group[group["anomaly_label"] == -1]
        anomaly_count += len(anomalies)
        all_results.append(group)
    
    print(f"  Analyzed {total_pairs} pairs → {anomaly_count} anomalies")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
def classify_severity(row):
    if row["anomaly_label"] != -1:
        return "✅ Normal"
    # Spatial bonus: if neighbors are also spiking AND this is anomalous → more likely real
    spatial_boost = False
    if "neighbor_spike" in row.index and row.get("neighbor_spike", 0) > 1.5:
        spatial_boost = True
    
    if row["anomaly_score"] < CRITICAL_SCORE_THRESHOLD or row["ratio_to_mean"] > CRITICAL_SPIKE_RATIO:
        return "🔴 CRITICAL"
    if spatial_boost:
        return "🔴 CRITICAL"  # Neighbor spread → escalate to critical
    return "⚠️ ALERT"
# ─────────────────────────────────────────────────────────────────────────────
# LOAD PINCODE COORDINATES
# ─────────────────────────────────────────────────────────────────────────────
def load_pincode_coords():
    """Load lat/long for pincodes from pincode_directory.csv."""
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
    
    # Standardize columns
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("pincode", "pin_code", "pin"):
            col_map[col] = "pincode"
        elif cl in ("latitude", "lat"):
            col_map[col] = "latitude"
        elif cl in ("longitude", "long", "lng", "lon"):
            col_map[col] = "longitude"
        elif cl in ("statename", "state_name", "state"):
            col_map[col] = "state"
    
    df = df.rename(columns=col_map)
    
    # Filter AP
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.upper().str.strip()
        df = df[df["state"].str.contains("ANDHRA", na=False)]
    
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    
    coords = (
        df.dropna(subset=["pincode", "latitude", "longitude"])
        [["pincode", "latitude", "longitude"]]
        .drop_duplicates(subset=["pincode"])
    )
    coords["pincode"] = coords["pincode"].astype(int)
    
    print(f"  Loaded coordinates for {len(coords)} pincodes")
    return coords
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
SPATIAL_FEATURE_COLS = [
    "case_count", "rolling_7d_mean", "rolling_7d_std", "rolling_14d_mean",
    "ratio_to_mean", "diff_1d", "day_of_week", "lag_7d_count", "cumulative_3d",
    "neighbor_cases", "spatial_ratio", "neighbor_spike"
]
STANDARD_FEATURE_COLS = [
    "case_count", "rolling_7d_mean", "rolling_7d_std", "rolling_14d_mean",
    "ratio_to_mean", "diff_1d", "day_of_week", "lag_7d_count", "cumulative_3d"
]
def main(ip_df=None, pincode_df=None):
    """
    Run Isolation Forest with adaptive contamination + spatial features.
    
    Pincode level uses spatial features (neighbor analysis).
    Mandal/district levels use standard features (no lat/long per mandal).
    """
    print_header("OUTBREAK DETECTION — ISOLATION FOREST (SPATIAL)")
    print(f"  Contamination: ADAPTIVE (min={MIN_CONTAMINATION}, max={MAX_CONTAMINATION})")
    print(f"  Neighbor radius: {RADIUS_KM} km")
    print(f"  Features: {len(SPATIAL_FEATURE_COLS)} (pincode) / {len(STANDARD_FEATURE_COLS)} (mandal/district)")
    
    # ── Load data ────────────────────────────────────────────────────────
    ip_df = load_ip_dataset(df=ip_df)
    pin_mapping, has_mandal = load_pincode_mapping(df=pincode_df)
    merged = merge_geography(ip_df, pin_mapping, has_mandal)
    
    # ── Load spatial data ────────────────────────────────────────────────
    print_header("SPATIAL SETUP")
    pincode_coords = load_pincode_coords()
    
    if pincode_coords is not None and len(pincode_coords) > 0:
        neighbor_map = build_neighbor_map(pincode_coords, RADIUS_KM)
        has_spatial = True
    else:
        print("  ⚠️ No pincode coordinates available. Running without spatial features.")
        neighbor_map = {}
        has_spatial = False
    
    # ── Pincode level (WITH spatial) ─────────────────────────────────────
    all_alerts = []
    
    print_header("SPATIAL ISOLATION FOREST — PINCODE LEVEL")
    daily_pincode = build_daily_series(merged, "pincode")
    
    if not daily_pincode.empty and has_spatial:
        # Build lookup for spatial features: (pincode, disease, date) → case_count
        daily_lookup = {}
        for _, row in daily_pincode.iterrows():
            key = (row["pincode"], row["complaint_name"], row["date"])
            daily_lookup[key] = row["case_count"]
        
        print_subheader("Building features with spatial neighbors")
        features = build_features_with_spatial(daily_pincode, neighbor_map, daily_lookup)
        
        if not features.empty:
            print(f"  Feature matrix: {len(features):,} rows × {len(SPATIAL_FEATURE_COLS)} features")
            
            print_subheader("Running Spatial Isolation Forest")
            results = detect_anomalies(features, "pincode", SPATIAL_FEATURE_COLS)
            
            if not results.empty:
                results["severity"] = results.apply(classify_severity, axis=1)
                results["level"] = "Pincode"
                anomalies = results[results["anomaly_label"] == -1].copy()
                
                if not anomalies.empty:
                    print_subheader("Top Spatial Anomalies — Pincode")
                    display = (
                        anomalies.sort_values("anomaly_score").head(30)
                        [["severity", "pincode", "complaint_name", "date",
                          "case_count", "ratio_to_mean", "neighbor_cases",
                          "spatial_ratio", "neighbor_spike", "anomaly_score"]]
                        .copy()
                    )
                    display["ratio_to_mean"] = display["ratio_to_mean"].round(1)
                    display["spatial_ratio"] = display["spatial_ratio"].round(1)
                    display["neighbor_spike"] = display["neighbor_spike"].round(1)
                    display["anomaly_score"] = display["anomaly_score"].round(3)
                    display.columns = [
                        "Severity", "Pincode", "Disease", "Date",
                        "Cases", "Spike", "Nbr_Cases", "Spatial_Ratio",
                        "Nbr_Spike", "Score"
                    ]
                    print(f"\n{display.to_string(index=False)}")
                    all_alerts.append(anomalies)
                else:
                    print("\n  ✅ No anomalies at Pincode level.")
    elif not daily_pincode.empty:
        # Fallback: no spatial, use standard features
        print("  Falling back to standard features (no spatial data).")
        features = build_features_standard(daily_pincode, "pincode")
        if not features.empty:
            results = detect_anomalies(features, "pincode", STANDARD_FEATURE_COLS)
            if not results.empty:
                results["severity"] = results.apply(classify_severity, axis=1)
                results["level"] = "Pincode"
                anomalies = results[results["anomaly_label"] == -1].copy()
                if not anomalies.empty:
                    all_alerts.append(anomalies)
    
    # ── Mandal & District levels (standard features) ─────────────────────
    other_levels = []
    if has_mandal:
        other_levels.append(("mandal", "Mandal"))
    other_levels.append(("district", "District"))
    
    for region_col, level_name in other_levels:
        print_header(f"ADAPTIVE ISOLATION FOREST — {level_name.upper()} LEVEL")
        
        daily = build_daily_series(merged, region_col)
        if daily.empty:
            continue
        
        print_subheader("Building features")
        features = build_features_standard(daily, region_col)
        if features.empty:
            continue
        print(f"  Feature matrix: {len(features):,} rows × {len(STANDARD_FEATURE_COLS)} features")
        
        print_subheader(f"Running Isolation Forest — {level_name}")
        results = detect_anomalies(features, region_col, STANDARD_FEATURE_COLS)
        
        if results.empty:
            continue
        
        results["severity"] = results.apply(classify_severity, axis=1)
        results["level"] = level_name
        anomalies = results[results["anomaly_label"] == -1].copy()
        
        if not anomalies.empty:
            print_subheader(f"Top Anomalies — {level_name}")
            display = (
                anomalies.sort_values("anomaly_score").head(20)
                [["severity", region_col, "complaint_name", "date",
                  "case_count", "ratio_to_mean", "anomaly_score"]]
                .copy()
            )
            display["ratio_to_mean"] = display["ratio_to_mean"].round(1)
            display["anomaly_score"] = display["anomaly_score"].round(3)
            print(f"\n{display.to_string(index=False)}")
            all_alerts.append(anomalies)
        else:
            print(f"\n  ✅ No anomalies at {level_name} level.")
    
    # ── Save ─────────────────────────────────────────────────────────────
    print_header("SAVING REPORT")
    if all_alerts:
        report = pd.concat(all_alerts, ignore_index=True)
        report = report.drop(columns=["_contamination", "_total_cases"], errors="ignore")
        report.to_csv(OUTPUT_PATH, index=False)
        print(f"\n  Saved {len(report)} anomalies to: {OUTPUT_PATH}")
    else:
        print("\n  No anomalies to save.")
    
    # ── Summary ──────────────────────────────────────────────────────────
    print_header("SUMMARY")
    total = sum(len(a) for a in all_alerts) if all_alerts else 0
    critical = sum((a["severity"] == "🔴 CRITICAL").sum() for a in all_alerts) if all_alerts else 0
    alert = total - critical
    print(f"""
  Total anomalies:  {total}
  🔴 CRITICAL:      {critical}
  ⚠️ ALERT:         {alert}
  
  Output: {OUTPUT_PATH}
  
  Improvements:
    ✅ Adaptive contamination (dynamic per pair)
    ✅ 12 features at pincode level (9 time-series + 3 spatial)
    ✅ Spatial: neighbor_cases, spatial_ratio, neighbor_spike
    ✅ Spatial boost: if neighbors spike too → escalate to CRITICAL
""")
if __name__ == "__main__":
    main()
