"""
Disease Outbreak Detection — Isolation Forest v2 (Fixed)
=========================================================
Fixes and improvements over previous versions:
  1. FIXED: ratio_to_mean denominator floored at 1.0 (avoids explosion near zero)
  2. FIXED: Added rolling_28d_mean baseline (captures monthly patterns)
  3. FIXED: Added ratio_7d_to_28d (short-term vs medium-term comparison)
  4. FIXED: Softened citations — features are INSPIRED BY classical epi methods
Feature design rationale:
  Features encode the signals prioritized in classical epidemiological
  surveillance methods (Hutwagner 2003 EARS, Farrington 1996) — short-term
  baseline comparison, variability, relative spike magnitude, acceleration,
  and day-of-week reporting patterns — as ML-compatible feature vectors.
  This follows the approach of modern ML-based surveillance systems
  (Wong et al. 2005, Generous et al. 2014, Fang et al. 2021).
  11 features per data point:
    Time-series (11):
      1.  case_count        raw daily count
      2.  rolling_7d_mean   short-term baseline (7 days, excl. today)
      3.  rolling_7d_std    short-term variability
      4.  rolling_28d_mean  medium-term baseline (28 days)
      5.  ratio_to_mean     case_count / max(7d_mean, 1.0) — floored
      6.  ratio_7d_to_28d   7d_mean / max(28d_mean, 1.0) — recent vs monthly
      7.  diff_1d           day-over-day change
      8.  day_of_week       0–6 (Mon–Sun)
      9.  lag_7d_count      cases exactly 7 days ago
      10. cumulative_3d     3-day rolling sum
      11. rolling_14d_mean  14-day baseline
    Spatial (3, pincode level only):
      12. neighbor_cases    total cases in pincodes within RADIUS_KM
      13. spatial_ratio     this pincode / neighbor average
      14. neighbor_spike    max spike ratio among neighbors
Usage:
  python outbreak_iforest_v2.py
  # From notebook:
  from outbreak_iforest_v2 import main
  main(ip_df=your_dataframe)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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
# Spatial
RADIUS_KM = 50
# Isolation Forest
N_ESTIMATORS = 200
RANDOM_STATE = 42
# Feature engineering
MIN_TOTAL_CASES = 20
MIN_DAYS = 14
# Ratio floor — prevents explosion when mean ≈ 0
# For rare diseases in small pincodes, mean can be 0.14 cases/day.
# Without floor: 2 / 0.14 = 14.3× (looks like massive spike, but it's 2 cases)
# With floor:    2 / 1.0  = 2.0  (sensible)
RATIO_FLOOR = 1.0
# Severity
CRITICAL_SCORE_THRESHOLD = -0.3
CRITICAL_SPIKE_RATIO = 5.0
# Output
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "outbreak_iforest_v2_report.csv")
# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def compute_contamination(total_cases):
    """Adaptive contamination: lower for high-volume, higher for low-volume."""
    raw = BASE_CASES / max(total_cases, 1)
    return max(MIN_CONTAMINATION, min(MAX_CONTAMINATION, raw))
def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))
def build_neighbor_map(pincode_coords, radius_km=RADIUS_KM):
    """Build pincode → list of neighbor pincodes within radius_km."""
    pincodes = pincode_coords["pincode"].values
    lats = pincode_coords["latitude"].values
    lons = pincode_coords["longitude"].values
    n = len(pincodes)
    print(f"  Building neighbor map for {n} pincodes (radius={radius_km}km)...")
    neighbors = {}
    deg_threshold = radius_km / 111.0 * 1.5
    for i in range(n):
        lat_mask = np.abs(lats - lats[i]) <= deg_threshold
        lon_mask = np.abs(lons - lons[i]) <= deg_threshold
        candidates = np.where(lat_mask & lon_mask)[0]
        nearby = []
        for j in candidates:
            if j != i and haversine_km(lats[i], lons[i], lats[j], lons[j]) <= radius_km:
                nearby.append(pincodes[j])
        neighbors[pincodes[i]] = nearby
    avg = np.mean([len(v) for v in neighbors.values()])
    print(f"  Avg neighbors per pincode: {avg:.1f}")
    return neighbors
def load_pincode_coords():
    """Load lat/long from pincode_directory.csv, filter AP."""
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
        if cl in ("pincode", "pin_code", "pin"): col_map[col] = "pincode"
        elif cl in ("latitude", "lat"): col_map[col] = "latitude"
        elif cl in ("longitude", "long", "lng", "lon"): col_map[col] = "longitude"
        elif cl in ("statename", "state_name", "state"): col_map[col] = "state"
    df = df.rename(columns=col_map)
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
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_time_features(group):
    """
    Add time-series features to a sorted group of daily case counts.
    Returns group with 11 feature columns added.
    """
    # Short-term baseline (7 days, excl. today)
    group["rolling_7d_mean"] = (
        group["case_count"].rolling(7, min_periods=3).mean().shift(1)
    )
    group["rolling_7d_std"] = (
        group["case_count"].rolling(7, min_periods=3).std().shift(1)
    )
    # Medium-term baseline (28 days — captures monthly patterns)
    # EARS uses 7 weekly periods; Farrington uses years.
    # 28 days is a practical middle ground for 6-month datasets.
    group["rolling_28d_mean"] = (
        group["case_count"].rolling(28, min_periods=10).mean().shift(1)
    )
    # 14-day baseline (bridging short and medium)
    group["rolling_14d_mean"] = (
        group["case_count"].rolling(14, min_periods=5).mean().shift(1)
    )
    # Spike ratio — FLOORED denominator to prevent explosion near zero
    # Without floor: 2 / 0.14 = 14.3× (misleading for 2 cases)
    # With floor:    2 / 1.0  = 2.0   (sensible)
    group["ratio_to_mean"] = group["case_count"] / group["rolling_7d_mean"].clip(lower=RATIO_FLOOR)
    # Short-term vs medium-term ratio — detects seasonal onsets
    # If 7d_mean >> 28d_mean, recent week is elevated vs the month
    group["ratio_7d_to_28d"] = (
        group["rolling_7d_mean"].fillna(0) /
        group["rolling_28d_mean"].clip(lower=RATIO_FLOOR)
    )
    # Day-over-day change (acceleration)
    group["diff_1d"] = group["case_count"].diff()
    # Day of week (reporting pattern: fewer on weekends)
    group["day_of_week"] = pd.to_datetime(group["date"]).dt.dayofweek
    # Same day last week
    group["lag_7d_count"] = group["case_count"].shift(7)
    # 3-day cumulative sum (sustained spike vs 1-day blip)
    group["cumulative_3d"] = group["case_count"].rolling(3, min_periods=1).sum()
    return group
TEMPORAL_FEATURE_COLS = [
    "case_count", "rolling_7d_mean", "rolling_7d_std", "rolling_14d_mean",
    "rolling_28d_mean", "ratio_to_mean", "ratio_7d_to_28d",
    "diff_1d", "day_of_week", "lag_7d_count", "cumulative_3d"
]
SPATIAL_EXTRA_COLS = ["neighbor_cases", "spatial_ratio", "neighbor_spike"]
REQUIRED_DROPNA = ["rolling_7d_mean", "rolling_7d_std", "diff_1d",
                    "rolling_28d_mean", "rolling_14d_mean", "lag_7d_count"]
def build_features(daily_df, region_col, neighbor_map=None, daily_lookup=None):
    """Build features. Adds spatial features if neighbor_map is provided and region_col='pincode'."""
    all_features = []
    use_spatial = (region_col == "pincode" and neighbor_map and daily_lookup)
    for (region, disease), group in daily_df.groupby([region_col, "complaint_name"]):
        group = group.sort_values("date").copy()
        total_cases = group["case_count"].sum()
        if total_cases < MIN_TOTAL_CASES or len(group) < MIN_DAYS:
            continue
        # Time-series features
        group = add_time_features(group)
        # Spatial features (pincode level only)
        if use_spatial:
            nbr_pincodes = neighbor_map.get(region, [])
            n_cases_list, spatial_ratio_list, nbr_spike_list = [], [], []
            for _, row in group.iterrows():
                date = row["date"]
                my_cases = row["case_count"]
                n_cases, n_count, max_ratio = 0, 0, 0.0
                for npc in nbr_pincodes:
                    nc = daily_lookup.get((npc, disease, date), 0)
                    if nc > 0:
                        n_cases += nc
                        n_count += 1
                        prev = daily_lookup.get((npc, disease,
                            pd.Timestamp(date) - pd.Timedelta(days=1)), 0)
                        if prev > 0:
                            max_ratio = max(max_ratio, nc / prev)
                nbr_avg = n_cases / max(n_count, 1)
                n_cases_list.append(n_cases)
                spatial_ratio_list.append(my_cases / max(nbr_avg, RATIO_FLOOR) if n_count > 0 else 1.0)
                nbr_spike_list.append(max_ratio)
            group["neighbor_cases"] = n_cases_list
            group["spatial_ratio"] = spatial_ratio_list
            group["neighbor_spike"] = nbr_spike_list
        # Drop rows where features aren't ready
        group = group.dropna(subset=REQUIRED_DROPNA)
        if len(group) < MIN_DAYS:
            continue
        group["_contamination"] = compute_contamination(total_cases)
        group["_total_cases"] = total_cases
        all_features.append(group)
    if all_features:
        result = pd.concat(all_features, ignore_index=True)
        n_pairs = result.groupby([region_col, "complaint_name"]).ngroups
        contams = result.groupby([region_col, "complaint_name"])["_contamination"].first()
        print(f"  Pairs: {n_pairs} | Contamination: {contams.min():.3f}–{contams.max():.3f}")
        return result
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies(features_df, region_col, feature_cols):
    """Run Isolation Forest with adaptive contamination per pair."""
    all_results = []
    anomaly_count = 0
    for (region, disease), group in features_df.groupby([region_col, "complaint_name"]):
        group = group.copy()
        contam = group["_contamination"].iloc[0]
        X = group[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = IsolationForest(
            n_estimators=N_ESTIMATORS, contamination=contam,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        group["anomaly_label"] = model.fit_predict(X_scaled)
        group["anomaly_score"] = model.decision_function(X_scaled)
        anomaly_count += (group["anomaly_label"] == -1).sum()
        all_results.append(group)
    print(f"  Total anomalies: {anomaly_count}")
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
def classify_severity(row):
    if row["anomaly_label"] != -1:
        return "✅ Normal"
    spatial_boost = ("neighbor_spike" in row.index and row.get("neighbor_spike", 0) > 1.5)
    if row["anomaly_score"] < CRITICAL_SCORE_THRESHOLD or row["ratio_to_mean"] > CRITICAL_SPIKE_RATIO:
        return "🔴 CRITICAL"
    if spatial_boost:
        return "🔴 CRITICAL"
    return "⚠️ ALERT"
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(ip_df=None, pincode_df=None):
    """
    Run fixed Isolation Forest with adaptive contamination + spatial features.
    Args:
        ip_df:      Pre-loaded health data DataFrame (e.g., from Impala).
        pincode_df: Pre-loaded pincode directory DataFrame.
    """
    print_header("OUTBREAK DETECTION — ISOLATION FOREST v2 (FIXED)")
    print(f"  Fixes applied:")
    print(f"    ✅ ratio_to_mean floored at {RATIO_FLOOR} (no explosion near zero)")
    print(f"    ✅ rolling_28d_mean + ratio_7d_to_28d (monthly baseline)")
    print(f"    ✅ Adaptive contamination ({MIN_CONTAMINATION}–{MAX_CONTAMINATION})")
    print(f"    ✅ Spatial features at pincode level ({RADIUS_KM}km radius)")
    # Load data
    ip_df = load_ip_dataset(df=ip_df)
    pin_mapping, has_mandal = load_pincode_mapping(df=pincode_df)
    merged = merge_geography(ip_df, pin_mapping, has_mandal)
    # Spatial setup
    print_subheader("Spatial setup")
    pincode_coords = load_pincode_coords()
    neighbor_map = build_neighbor_map(pincode_coords) if pincode_coords is not None else {}
    has_spatial = bool(neighbor_map)
    all_alerts = []
    # ── Pincode level (with spatial) ─────────────────────────────────────
    print_header("IF v2 — PINCODE LEVEL")
    daily_pin = build_daily_series(merged, "pincode")
    if not daily_pin.empty:
        daily_lookup = {
            (r["pincode"], r["complaint_name"], r["date"]): r["case_count"]
            for _, r in daily_pin.iterrows()
        } if has_spatial else None
        print_subheader("Building features (11 temporal + 3 spatial)")
        features = build_features(daily_pin, "pincode", neighbor_map, daily_lookup)
        if not features.empty:
            feat_cols = TEMPORAL_FEATURE_COLS + (SPATIAL_EXTRA_COLS if has_spatial else [])
            print(f"  Matrix: {len(features):,} × {len(feat_cols)} features")
            print_subheader("Running IF v2")
            results = detect_anomalies(features, "pincode", feat_cols)
            if not results.empty:
                results["severity"] = results.apply(classify_severity, axis=1)
                results["level"] = "Pincode"
                anomalies = results[results["anomaly_label"] == -1].copy()
                if not anomalies.empty:
                    all_alerts.append(anomalies)
                    _print_top(anomalies, "pincode", has_spatial)
    # ── Mandal & District levels (no spatial) ────────────────────────────
    other_levels = []
    if has_mandal:
        other_levels.append(("mandal", "Mandal"))
    other_levels.append(("district", "District"))
    for region_col, level_name in other_levels:
        print_header(f"IF v2 — {level_name.upper()} LEVEL")
        daily = build_daily_series(merged, region_col)
        if daily.empty:
            continue
        print_subheader("Building features (11 temporal)")
        features = build_features(daily, region_col)
        if features.empty:
            continue
        print(f"  Matrix: {len(features):,} × {len(TEMPORAL_FEATURE_COLS)} features")
        print_subheader(f"Running IF v2 — {level_name}")
        results = detect_anomalies(features, region_col, TEMPORAL_FEATURE_COLS)
        if results.empty:
            continue
        results["severity"] = results.apply(classify_severity, axis=1)
        results["level"] = level_name
        anomalies = results[results["anomaly_label"] == -1].copy()
        if not anomalies.empty:
            all_alerts.append(anomalies)
            _print_top(anomalies, region_col, False)
        else:
            print(f"\n  ✅ No anomalies at {level_name} level.")
    # ── Save ─────────────────────────────────────────────────────────────
    print_header("REPORT")
    if all_alerts:
        report = pd.concat(all_alerts, ignore_index=True)
        report = report.drop(columns=["_contamination", "_total_cases"], errors="ignore")
        report.to_csv(OUTPUT_PATH, index=False)
        total = len(report)
        critical = (report["severity"] == "🔴 CRITICAL").sum()
        print(f"  Saved {total} anomalies ({critical} critical) to: {OUTPUT_PATH}")
    else:
        print("  No anomalies to save.")
def _print_top(anomalies, region_col, has_spatial):
    """Print top anomalies."""
    cols = ["severity", region_col, "complaint_name", "date",
            "case_count", "ratio_to_mean", "ratio_7d_to_28d", "anomaly_score"]
    if has_spatial:
        cols += ["neighbor_cases", "spatial_ratio"]
    display = anomalies.sort_values("anomaly_score").head(20)[cols].copy()
    for c in ["ratio_to_mean", "ratio_7d_to_28d", "spatial_ratio"]:
        if c in display.columns:
            display[c] = display[c].round(1)
    display["anomaly_score"] = display["anomaly_score"].round(3)
    print(f"\n{display.to_string(index=False)}")
if __name__ == "__main__":
    main()
