"""
Disease Outbreak Detection — Isolation Forest with Epidemiological Metrics
===========================================================================
Builds on IF v2 (all fixes applied) and adds three epidemiological layers:
  1. Rₜ (Effective Reproduction Number)
     Estimates how many people one case infects RIGHT NOW.
     Rₜ > 1 → growing, Rₜ = 1 → stable, Rₜ < 1 → declining.
     Method: Cori et al. (2013), simplified ratio estimator.
  2. Incidence Rate (district level only)
     Cases per 100,000 population per week.
     Normalizes for district population so comparisons are fair.
     Uses AP district populations (Census 2011 projected to 2025).
     NOT computed at pincode level — no reliable population data.
  3. Epidemic Curve Classification
     Automatically classifies outbreak shape:
       - Point Source:  single sharp spike, rapid decline (food poisoning)
       - Propagated:    successive waves (person-to-person transmission)
       - Continuous:    sustained elevation (contaminated water, endemic)
       - Sporadic:      isolated cases, no clear pattern
Usage:
  python outbreak_iforest_epi.py
  # From notebook:
  from outbreak_iforest_epi import main
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
RATIO_FLOOR = 1.0
# Rₜ estimation
RT_GENERATION_INTERVAL = 7   # Days (serial interval proxy)
                              # Dengue ~10d, respiratory ~5d, generic default = 7d
RT_SMOOTHING_WINDOW = 7      # Smooth Rₜ to reduce noise
# Severity
CRITICAL_SCORE_THRESHOLD = -0.3
CRITICAL_SPIKE_RATIO = 5.0
# Output
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "outbreak_iforest_epi_report.csv")
RT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "rt_estimates.csv")
INCIDENCE_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "incidence_rates.csv")
EPI_CURVE_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "epidemic_curves.csv")
# ─────────────────────────────────────────────────────────────────────────────
# AP DISTRICT POPULATIONS (Census 2011 → projected 2025 at ~1.1%/yr growth)
# Post-2014 bifurcation: 13 districts in Andhra Pradesh
# Source: Census of India 2011 + Registrar General projections
# These are approximate. Replace with actual data if available.
# ─────────────────────────────────────────────────────────────────────────────
AP_DISTRICT_POPULATION = {
    "Anantapur":        4_360_000,
    "Chittoor":         4_530_000,
    "East Godavari":    5_560_000,
    "Guntur":           5_150_000,
    "Krishna":          4_850_000,
    "Kurnool":          4_300_000,
    "Nellore":          3_200_000,   # Sri Potti Sriramulu Nellore
    "Prakasam":         3_620_000,
    "Srikakulam":       2_930_000,
    "Visakhapatnam":    4_640_000,
    "Vizianagaram":     2_480_000,
    "West Godavari":    4_100_000,
    "Y.S.R. Kadapa":    3_050_000,   # YSR / Kadapa
    # Common alternate names
    "Kadapa":           3_050_000,
    "Ysr Kadapa":       3_050_000,
    "Ysr":              3_050_000,
    "Sri Potti Sriramulu Nellore": 3_200_000,
    "Spsr Nellore":     3_200_000,
}
def get_district_population(district_name):
    """Look up district population with fuzzy matching."""
    name = str(district_name).strip().title()
    if name in AP_DISTRICT_POPULATION:
        return AP_DISTRICT_POPULATION[name]
    # Try partial match
    for key, pop in AP_DISTRICT_POPULATION.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return pop
    return None
# ─────────────────────────────────────────────────────────────────────────────
# 1. Rₜ ESTIMATION (Cori Method, Simplified)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_rt(daily_cases, generation_interval=RT_GENERATION_INTERVAL,
                smoothing=RT_SMOOTHING_WINDOW):
    """
    Estimate Rₜ using the simplified Cori method.
    Rₜ(t) = Σ cases[t-τ+1 : t] / Σ cases[t-2τ+1 : t-τ]
    Where τ = generation_interval (default 7 days).
    This is equivalent to: "ratio of cases in the last generation
    to cases in the previous generation."
    Returns a Series of Rₜ values indexed by date.
    Reference:
      Cori et al. (2013). "A New Framework and Software to Estimate
      Time-Varying Reproduction Numbers During Epidemics."
      American Journal of Epidemiology, 178(9), 1505-1512.
    """
    cases = daily_cases.sort_values("date").copy()
    cases = cases.set_index("date")["case_count"]
    tau = generation_interval
    # Rolling sums for numerator and denominator
    numerator = cases.rolling(window=tau, min_periods=tau).sum()
    denominator = cases.shift(tau).rolling(window=tau, min_periods=tau).sum()
    # Rₜ = recent generation / previous generation
    rt = numerator / denominator.clip(lower=1)  # Floor at 1 to avoid division by zero
    # Smooth to reduce noise
    rt_smooth = rt.rolling(window=smoothing, min_periods=3).mean()
    return rt_smooth
def compute_rt_all(merged_df, region_col="district"):
    """
    Compute Rₜ for each (disease, region) pair.
    Returns a DataFrame with date, region, disease, Rₜ.
    """
    print_subheader("Computing Rₜ (Effective Reproduction Number)")
    print(f"  Generation interval: {RT_GENERATION_INTERVAL} days")
    print(f"  Smoothing window: {RT_SMOOTHING_WINDOW} days")
    daily = build_daily_series(merged_df, region_col)
    if daily.empty:
        return pd.DataFrame()
    all_rt = []
    for (region, disease), group in daily.groupby([region_col, "complaint_name"]):
        if group["case_count"].sum() < MIN_TOTAL_CASES or len(group) < MIN_DAYS * 2:
            continue
        rt_series = estimate_rt(group)
        rt_df = pd.DataFrame({
            "date": rt_series.index,
            "rt": rt_series.values,
            region_col: region,
            "complaint_name": disease
        }).dropna()
        all_rt.append(rt_df)
    if all_rt:
        result = pd.concat(all_rt, ignore_index=True)
        # Stats
        latest_rt = result.groupby([region_col, "complaint_name"])["rt"].last()
        growing = (latest_rt > 1.0).sum()
        declining = (latest_rt < 1.0).sum()
        print(f"  Pairs analyzed: {len(latest_rt)}")
        print(f"  Currently growing (Rₜ > 1): {growing}")
        print(f"  Currently declining (Rₜ < 1): {declining}")
        # Flag high Rₜ
        high_rt = latest_rt[latest_rt > 1.5].sort_values(ascending=False).head(10)
        if len(high_rt) > 0:
            print(f"\n  ⚠️ High Rₜ (spreading fast):")
            for (region, disease), rt_val in high_rt.items():
                print(f"    Rₜ={rt_val:.2f}  {disease} in {region}")
        return result
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# 2. INCIDENCE RATE (District Level Only)
# ─────────────────────────────────────────────────────────────────────────────
def compute_incidence_rates(merged_df):
    """
    Compute weekly incidence rate per 100,000 population at district level.
    Incidence Rate = (cases in period / population) × 100,000
    Only computed at DISTRICT level since pincode-level population
    data is not reliably available.
    """
    print_subheader("Computing Incidence Rates (District Level)")
    if "district" not in merged_df.columns:
        print("  ⚠️ No district column. Skipping incidence rate.")
        return pd.DataFrame()
    df = merged_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    # Weekly case counts per district per disease
    weekly = (
        df.groupby(["district", "complaint_name", "year", "week"])
        .agg(weekly_cases=("health_id", "count"))
        .reset_index()
    )
    # Add population and compute incidence rate
    weekly["population"] = weekly["district"].apply(get_district_population)
    matched = weekly["population"].notna().sum()
    total = len(weekly)
    print(f"  Population match: {matched}/{total} rows "
          f"({weekly['district'].nunique()} districts)")
    unmapped = weekly[weekly["population"].isna()]["district"].unique()
    if len(unmapped) > 0:
        print(f"  ⚠️ No population for: {list(unmapped)[:10]}")
    weekly = weekly.dropna(subset=["population"])
    weekly["incidence_rate"] = (weekly["weekly_cases"] / weekly["population"]) * 100_000
    if len(weekly) > 0:
        # Top incidence rates
        latest_week = weekly["week"].max()
        latest = weekly[weekly["week"] == latest_week].sort_values(
            "incidence_rate", ascending=False
        ).head(10)
        if len(latest) > 0:
            print(f"\n  Top incidence rates (week {latest_week}):")
            for _, row in latest.iterrows():
                print(f"    {row['incidence_rate']:.1f}/100K  "
                      f"{row['complaint_name']} in {row['district']} "
                      f"({row['weekly_cases']} cases)")
    return weekly
# ─────────────────────────────────────────────────────────────────────────────
# 3. EPIDEMIC CURVE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def classify_epidemic_curve(daily_cases):
    """
    Classify the shape of an outbreak's epidemic curve.
    Categories:
      - Point Source:  single peak, symmetric, duration < 3 weeks
                       (food poisoning, single exposure event)
      - Propagated:    multiple peaks spaced ~7 days apart
                       (person-to-person transmission)
      - Continuous:    sustained elevation above baseline for >3 weeks
                       (ongoing environmental exposure)
      - Sporadic:      low, irregular cases, no clear pattern
    Method: analyze peak structure, duration, and pattern of the case time series.
    """
    cases = daily_cases.values if hasattr(daily_cases, "values") else np.array(daily_cases)
    if len(cases) < 7 or np.sum(cases) < 5:
        return "Sporadic"
    # Smooth for peak detection
    if len(cases) >= 7:
        smooth = pd.Series(cases).rolling(3, min_periods=1, center=True).mean().values
    else:
        smooth = cases.astype(float)
    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > 1:
            peaks.append(i)
    # Duration above baseline
    baseline = np.percentile(cases, 25)
    above_baseline = cases > max(baseline + 1, 2)
    elevated_days = np.sum(above_baseline)
    # Maximum case count
    max_cases = np.max(cases)
    mean_cases = np.mean(cases)
    # Coefficient of variation (variability relative to mean)
    cv = np.std(cases) / max(mean_cases, 0.1)
    # ── Classification logic ──────────────────────────────────────────
    # Sporadic: very low, irregular
    if max_cases <= 3 or elevated_days < 3:
        return "Sporadic"
    # Point Source: single peak, short duration
    if len(peaks) <= 2 and elevated_days <= 21:
        # Check for symmetry around peak
        if len(peaks) >= 1:
            peak_idx = peaks[0]
            pre_peak = cases[:peak_idx]
            post_peak = cases[peak_idx:]
            if len(pre_peak) > 0 and len(post_peak) > 0:
                # Rising then falling
                if max_cases > 3 * mean_cases:
                    return "Point Source"
    # Propagated: multiple distinct peaks
    if len(peaks) >= 2:
        # Check if peaks are spaced roughly evenly (generation intervals)
        peak_gaps = np.diff(peaks)
        if len(peak_gaps) > 0:
            mean_gap = np.mean(peak_gaps)
            if 4 <= mean_gap <= 21:  # 4–21 days between peaks
                return "Propagated"
    # Continuous: sustained elevation
    if elevated_days >= 21:
        # Long period above baseline without clear peak structure
        if cv < 1.5:  # Relatively steady
            return "Continuous"
    # Propagated catch-all: multiple peaks with high variability
    if len(peaks) >= 3 and cv > 0.8:
        return "Propagated"
    # Continuous catch-all
    if elevated_days >= 14:
        return "Continuous"
    # Default
    if len(peaks) == 1 and elevated_days <= 14:
        return "Point Source"
    return "Sporadic"
def compute_epidemic_curves(merged_df, region_col="district"):
    """
    Classify epidemic curves for each (disease, region) pair.
    """
    print_subheader("Classifying Epidemic Curves")
    daily = build_daily_series(merged_df, region_col)
    if daily.empty:
        return pd.DataFrame()
    results = []
    for (region, disease), group in daily.groupby([region_col, "complaint_name"]):
        total = group["case_count"].sum()
        if total < 10 or len(group) < 14:
            continue
        curve_type = classify_epidemic_curve(group.sort_values("date")["case_count"])
        results.append({
            region_col: region,
            "complaint_name": disease,
            "curve_type": curve_type,
            "total_cases": total,
            "duration_days": len(group),
            "peak_cases": group["case_count"].max(),
            "mean_cases": round(group["case_count"].mean(), 1),
        })
    if results:
        result_df = pd.DataFrame(results)
        # Summary
        for ct in ["Point Source", "Propagated", "Continuous", "Sporadic"]:
            count = (result_df["curve_type"] == ct).sum()
            if count > 0:
                print(f"  {ct}: {count} outbreak(s)")
        return result_df
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# IF DETECTION (same as v2)
# ─────────────────────────────────────────────────────────────────────────────
def compute_contamination(total_cases):
    raw = BASE_CASES / max(total_cases, 1)
    return max(MIN_CONTAMINATION, min(MAX_CONTAMINATION, raw))
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))
def build_neighbor_map(pincode_coords, radius_km=RADIUS_KM):
    pincodes = pincode_coords["pincode"].values
    lats = pincode_coords["latitude"].values
    lons = pincode_coords["longitude"].values
    n = len(pincodes)
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
    return neighbors
def load_pincode_coords():
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
    return coords
def add_time_features(group):
    group["rolling_7d_mean"] = group["case_count"].rolling(7, min_periods=3).mean().shift(1)
    group["rolling_7d_std"] = group["case_count"].rolling(7, min_periods=3).std().shift(1)
    group["rolling_28d_mean"] = group["case_count"].rolling(28, min_periods=10).mean().shift(1)
    group["rolling_14d_mean"] = group["case_count"].rolling(14, min_periods=5).mean().shift(1)
    group["ratio_to_mean"] = group["case_count"] / group["rolling_7d_mean"].clip(lower=RATIO_FLOOR)
    group["ratio_7d_to_28d"] = (
        group["rolling_7d_mean"].fillna(0) / group["rolling_28d_mean"].clip(lower=RATIO_FLOOR)
    )
    group["diff_1d"] = group["case_count"].diff()
    group["day_of_week"] = pd.to_datetime(group["date"]).dt.dayofweek
    group["lag_7d_count"] = group["case_count"].shift(7)
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
    all_features = []
    use_spatial = (region_col == "pincode" and neighbor_map and daily_lookup)
    for (region, disease), group in daily_df.groupby([region_col, "complaint_name"]):
        group = group.sort_values("date").copy()
        total_cases = group["case_count"].sum()
        if total_cases < MIN_TOTAL_CASES or len(group) < MIN_DAYS:
            continue
        group = add_time_features(group)
        if use_spatial:
            nbr_pincodes = neighbor_map.get(region, [])
            n_cases_list, sr_list, ns_list = [], [], []
            for _, row in group.iterrows():
                date, my_cases = row["date"], row["case_count"]
                n_cases, n_count, max_r = 0, 0, 0.0
                for npc in nbr_pincodes:
                    nc = daily_lookup.get((npc, disease, date), 0)
                    if nc > 0:
                        n_cases += nc; n_count += 1
                        prev = daily_lookup.get((npc, disease,
                            pd.Timestamp(date) - pd.Timedelta(days=1)), 0)
                        if prev > 0: max_r = max(max_r, nc / prev)
                nbr_avg = n_cases / max(n_count, 1)
                n_cases_list.append(n_cases)
                sr_list.append(my_cases / max(nbr_avg, RATIO_FLOOR) if n_count > 0 else 1.0)
                ns_list.append(max_r)
            group["neighbor_cases"] = n_cases_list
            group["spatial_ratio"] = sr_list
            group["neighbor_spike"] = ns_list
        group = group.dropna(subset=REQUIRED_DROPNA)
        if len(group) < MIN_DAYS:
            continue
        group["_contamination"] = compute_contamination(total_cases)
        group["_total_cases"] = total_cases
        all_features.append(group)
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
def detect_anomalies(features_df, region_col, feature_cols):
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
    Run Isolation Forest v2 + epidemiological metrics.
    Outputs:
      - outbreak_iforest_epi_report.csv    (IF anomalies)
      - rt_estimates.csv                    (Rₜ per disease per district)
      - incidence_rates.csv                 (weekly incidence per 100K)
      - epidemic_curves.csv                 (curve classification)
    """
    print_header("OUTBREAK DETECTION — IF v2 + EPIDEMIOLOGICAL METRICS")
    print(f"  IF: Adaptive contamination + spatial features + all fixes")
    print(f"  Epi: Rₜ + Incidence Rate (district) + Epidemic Curve")
    # ── Load data ────────────────────────────────────────────────────────
    ip_df = load_ip_dataset(df=ip_df)
    pin_mapping, has_mandal = load_pincode_mapping(df=pincode_df)
    merged = merge_geography(ip_df, pin_mapping, has_mandal)
    # ── Spatial setup ────────────────────────────────────────────────────
    pincode_coords = load_pincode_coords()
    neighbor_map = build_neighbor_map(pincode_coords) if pincode_coords is not None else {}
    has_spatial = bool(neighbor_map)
    # ======================================================================
    # PART A: ISOLATION FOREST DETECTION (same as v2)
    # ======================================================================
    print_header("PART A: ISOLATION FOREST ANOMALY DETECTION")
    all_alerts = []
    # Pincode level (with spatial)
    print_subheader("Pincode Level")
    daily_pin = build_daily_series(merged, "pincode")
    if not daily_pin.empty:
        daily_lookup = {
            (r["pincode"], r["complaint_name"], r["date"]): r["case_count"]
            for _, r in daily_pin.iterrows()
        } if has_spatial else None
        features = build_features(daily_pin, "pincode", neighbor_map, daily_lookup)
        if not features.empty:
            feat_cols = TEMPORAL_FEATURE_COLS + (SPATIAL_EXTRA_COLS if has_spatial else [])
            results = detect_anomalies(features, "pincode", feat_cols)
            if not results.empty:
                results["severity"] = results.apply(classify_severity, axis=1)
                results["level"] = "Pincode"
                anomalies = results[results["anomaly_label"] == -1].copy()
                if not anomalies.empty:
                    all_alerts.append(anomalies)
    # Mandal & District levels
    other_levels = []
    if has_mandal:
        other_levels.append(("mandal", "Mandal"))
    other_levels.append(("district", "District"))
    for region_col, level_name in other_levels:
        print_subheader(f"{level_name} Level")
        daily = build_daily_series(merged, region_col)
        if daily.empty:
            continue
        features = build_features(daily, region_col)
        if features.empty:
            continue
        results = detect_anomalies(features, region_col, TEMPORAL_FEATURE_COLS)
        if results.empty:
            continue
        results["severity"] = results.apply(classify_severity, axis=1)
        results["level"] = level_name
        anomalies = results[results["anomaly_label"] == -1].copy()
        if not anomalies.empty:
            all_alerts.append(anomalies)
    # Save IF report
    if all_alerts:
        report = pd.concat(all_alerts, ignore_index=True)
        report = report.drop(columns=["_contamination", "_total_cases"], errors="ignore")
        report.to_csv(OUTPUT_PATH, index=False)
        total_if = len(report)
        critical_if = (report["severity"] == "🔴 CRITICAL").sum()
        print(f"\n  IF Report: {total_if} anomalies ({critical_if} critical) → {OUTPUT_PATH}")
    else:
        total_if, critical_if = 0, 0
        print("\n  No IF anomalies detected.")
    # ======================================================================
    # PART B: Rₜ ESTIMATION
    # ======================================================================
    print_header("PART B: Rₜ (EFFECTIVE REPRODUCTION NUMBER)")
    rt_df = compute_rt_all(merged, region_col="district")
    if not rt_df.empty:
        rt_df.to_csv(RT_OUTPUT_PATH, index=False)
        print(f"\n  Saved Rₜ estimates to: {RT_OUTPUT_PATH}")
    # ======================================================================
    # PART C: INCIDENCE RATE (District Level Only)
    # ======================================================================
    print_header("PART C: INCIDENCE RATE (per 100K)")
    incidence_df = compute_incidence_rates(merged)
    if not incidence_df.empty:
        incidence_df.to_csv(INCIDENCE_OUTPUT_PATH, index=False)
        print(f"\n  Saved incidence rates to: {INCIDENCE_OUTPUT_PATH}")
    # ======================================================================
    # PART D: EPIDEMIC CURVE CLASSIFICATION
    # ======================================================================
    print_header("PART D: EPIDEMIC CURVE CLASSIFICATION")
    epi_curves = compute_epidemic_curves(merged, region_col="district")
    if not epi_curves.empty:
        epi_curves.to_csv(EPI_CURVE_OUTPUT_PATH, index=False)
        print(f"\n  Saved epidemic curves to: {EPI_CURVE_OUTPUT_PATH}")
        # Show propagated outbreaks (most concerning)
        propagated = epi_curves[epi_curves["curve_type"] == "Propagated"]
        if not propagated.empty:
            print(f"\n  ⚠️ Propagated outbreaks (person-to-person spread):")
            for _, row in propagated.sort_values("total_cases", ascending=False).head(10).iterrows():
                print(f"    {row['complaint_name']} in {row['district']} "
                      f"— {row['total_cases']} cases, peak {row['peak_cases']}/day")
    # ======================================================================
    # SUMMARY
    # ======================================================================
    print_header("FULL SUMMARY")
    # Combine IF alerts with Rₜ and curve info
    if all_alerts and not rt_df.empty:
        # Get latest Rₜ per disease-district
        latest_rt = (
            rt_df.sort_values("date")
            .groupby(["district", "complaint_name"])["rt"]
            .last()
            .reset_index()
        )
        # Cross-reference: anomalies where Rₜ > 1 are MOST concerning
        if "district" in report.columns:
            enriched = report.merge(
                latest_rt, on=["district", "complaint_name"], how="left"
            )
            high_priority = enriched[
                (enriched["severity"] == "🔴 CRITICAL") &
                (enriched["rt"] > 1.0)
            ]
            if not high_priority.empty:
                print(f"\n  🚨 HIGH PRIORITY: {len(high_priority)} alerts where BOTH:")
                print(f"     • IF flagged as CRITICAL")
                print(f"     • Rₜ > 1.0 (actively spreading)")
                for disease in high_priority["complaint_name"].unique()[:5]:
                    subset = high_priority[high_priority["complaint_name"] == disease]
                    districts = ", ".join(subset["district"].unique()[:3])
                    avg_rt = subset["rt"].mean()
                    print(f"     → {disease} (Rₜ={avg_rt:.2f}) in {districts}")
    print(f"""
  Outputs:
    📊 IF Anomalies:     {OUTPUT_PATH}
    📈 Rₜ Estimates:     {RT_OUTPUT_PATH}
    🏥 Incidence Rates:  {INCIDENCE_OUTPUT_PATH}
    📉 Epidemic Curves:  {EPI_CURVE_OUTPUT_PATH}
  Interpretation:
    • IF anomalies:     statistical outliers in case patterns
    • Rₜ > 1:           disease is actively spreading
    • High incidence:   many cases per capita (population-normalized)
    • Propagated curve: person-to-person transmission pattern
    🚨 HIGHEST CONCERN: IF critical + Rₜ > 1 + propagated curve
""")
if __name__ == "__main__":
    main()
