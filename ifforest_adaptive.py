"""
Disease Outbreak Detection — Isolation Forest with Adaptive Contamination
==========================================================================
Improves on the base Isolation Forest by dynamically adjusting the
contamination parameter per (disease, region) pair based on data volume.
Key improvement:
  - Low-volume pairs (few total cases) → higher contamination (more sensitive)
  - High-volume pairs (many total cases) → lower contamination (fewer false alerts)
  - Formula: contamination = clamp(BASE_CASES / total_cases, MIN_CONT, MAX_CONT)
Also adds extra features:
  - rolling_14d_mean: longer baseline context
  - lag_7d_count:     same-day-last-week count (weekly cycles)
  - cumulative_3d:    3-day rolling sum (sustained spike detection)
Usage:
  # Standalone (reads ip.csv):
  python outbreak_iforest_adaptive.py
  # From notebook with Impala DataFrame:
  from outbreak_iforest_adaptive import main
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
# Adaptive contamination parameters
MIN_CONTAMINATION = 0.01      # Floor: at most 1% for high-volume pairs
MAX_CONTAMINATION = 0.10      # Ceiling: at most 10% for low-volume pairs
BASE_CASES = 50               # Pairs with 50 total cases → ~1% contamination
                              # Pairs with 25 total cases → ~2%
                              # Pairs with 500 total cases → 0.1% (clamped to MIN)
# Isolation Forest
N_ESTIMATORS = 200
RANDOM_STATE = 42
# Feature engineering
ROLLING_WINDOW = 7
MIN_TOTAL_CASES = 20          # Skip very low volume pairs
MIN_DAYS = 14
# Severity thresholds
CRITICAL_SCORE_THRESHOLD = -0.3
CRITICAL_SPIKE_RATIO = 5.0
# Output
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "outbreak_iforest_adaptive_report.csv")
# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE CONTAMINATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_contamination(total_cases):
    """
    Adaptively compute contamination based on total case volume.
    
    Logic:
      - Low-volume (20 cases):  50/20  = 2.5% → more sensitive
      - Medium (100 cases):     50/100 = 0.5% → balanced
      - High-volume (1000):     50/1000= 0.05% → clamped to 1%
    
    This prevents high-volume diseases from generating thousands of alerts
    while keeping sensitivity for rare diseases.
    """
    raw = BASE_CASES / max(total_cases, 1)
    return max(MIN_CONTAMINATION, min(MAX_CONTAMINATION, raw))
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(daily_df, region_col):
    """
    Build enhanced features for Isolation Forest.
    
    9 features per day (vs 6 in base version):
      1. case_count         raw count
      2. rolling_7d_mean    7-day rolling mean (excl. today)
      3. rolling_7d_std     7-day rolling std
      4. rolling_14d_mean   14-day rolling mean (longer baseline) [NEW]
      5. ratio_to_mean      case_count / 7d mean
      6. diff_1d            change from yesterday
      7. day_of_week        0-6
      8. lag_7d_count       cases exactly 7 days ago [NEW]
      9. cumulative_3d      sum of past 3 days (sustained spike) [NEW]
    """
    all_features = []
    pair_stats = []
    
    for (region, disease), group in daily_df.groupby([region_col, "complaint_name"]):
        group = group.sort_values("date").copy()
        
        total_cases = group["case_count"].sum()
        if total_cases < MIN_TOTAL_CASES or len(group) < MIN_DAYS:
            continue
        
        # ── Standard features ──────────────────────────────────────────
        group["rolling_7d_mean"] = (
            group["case_count"]
            .rolling(window=ROLLING_WINDOW, min_periods=3)
            .mean()
            .shift(1)
        )
        group["rolling_7d_std"] = (
            group["case_count"]
            .rolling(window=ROLLING_WINDOW, min_periods=3)
            .std()
            .shift(1)
        )
        group["ratio_to_mean"] = np.where(
            group["rolling_7d_mean"] > 0,
            group["case_count"] / group["rolling_7d_mean"],
            group["case_count"]
        )
        group["diff_1d"] = group["case_count"].diff()
        group["day_of_week"] = pd.to_datetime(group["date"]).dt.dayofweek
        
        # ── New features ───────────────────────────────────────────────
        # 14-day rolling mean (longer baseline)
        group["rolling_14d_mean"] = (
            group["case_count"]
            .rolling(window=14, min_periods=5)
            .mean()
            .shift(1)
        )
        
        # Lag-7: same day last week
        group["lag_7d_count"] = group["case_count"].shift(7)
        
        # Cumulative 3-day sum (captures sustained spikes)
        group["cumulative_3d"] = (
            group["case_count"]
            .rolling(window=3, min_periods=1)
            .sum()
        )
        
        # Drop rows where features aren't ready
        group = group.dropna(subset=[
            "rolling_7d_mean", "rolling_7d_std", "diff_1d",
            "rolling_14d_mean", "lag_7d_count"
        ])
        
        if len(group) < MIN_DAYS:
            continue
        
        # Compute adaptive contamination for this pair
        contam = compute_contamination(total_cases)
        group["_contamination"] = contam
        group["_total_cases"] = total_cases
        
        all_features.append(group)
        pair_stats.append({
            "region": region, "disease": disease,
            "total_cases": total_cases, "days": len(group),
            "contamination": contam
        })
    
    if pair_stats:
        stats_df = pd.DataFrame(pair_stats)
        print(f"  Pairs to analyze: {len(stats_df)}")
        print(f"  Contamination range: {stats_df['contamination'].min():.3f} — "
              f"{stats_df['contamination'].max():.3f}")
        print(f"  Avg contamination: {stats_df['contamination'].mean():.3f}")
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST WITH ADAPTIVE CONTAMINATION
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "case_count", "rolling_7d_mean", "rolling_7d_std", "rolling_14d_mean",
    "ratio_to_mean", "diff_1d", "day_of_week", "lag_7d_count", "cumulative_3d"
]
def detect_anomalies(features_df, region_col):
    """
    Run Isolation Forest per (disease, region) pair WITH adaptive contamination.
    Each pair gets its own contamination value based on total case volume.
    """
    all_results = []
    anomaly_count = 0
    total_pairs = 0
    
    for (region, disease), group in features_df.groupby([region_col, "complaint_name"]):
        total_pairs += 1
        group = group.copy()
        
        # Get adaptive contamination for this pair
        contam = group["_contamination"].iloc[0]
        
        X = group[FEATURE_COLS].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit with adaptive contamination
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
    
    print(f"  Analyzed {total_pairs} pairs")
    print(f"  Total anomalies: {anomaly_count}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
def classify_severity(row):
    """Classify anomaly severity."""
    if row["anomaly_label"] != -1:
        return "✅ Normal"
    if row["anomaly_score"] < CRITICAL_SCORE_THRESHOLD or row["ratio_to_mean"] > CRITICAL_SPIKE_RATIO:
        return "🔴 CRITICAL"
    return "⚠️ ALERT"
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(ip_df=None, pincode_df=None):
    """
    Run Isolation Forest with adaptive contamination.
    
    Args:
        ip_df:      Pre-loaded health data DataFrame (e.g., from Impala).
        pincode_df: Pre-loaded pincode directory DataFrame.
    """
    print_header("OUTBREAK DETECTION — ISOLATION FOREST (ADAPTIVE)")
    print(f"  Contamination: ADAPTIVE (min={MIN_CONTAMINATION}, max={MAX_CONTAMINATION})")
    print(f"  Base cases for scaling: {BASE_CASES}")
    print(f"  Trees: {N_ESTIMATORS}")
    print(f"  Features: {len(FEATURE_COLS)} (enhanced)")
    
    # ── Load data ────────────────────────────────────────────────────────
    ip_df = load_ip_dataset(df=ip_df)
    pin_mapping, has_mandal = load_pincode_mapping(df=pincode_df)
    merged = merge_geography(ip_df, pin_mapping, has_mandal)
    
    # ── Run at each level ────────────────────────────────────────────────
    levels = [("pincode", "Pincode")]
    if has_mandal:
        levels.append(("mandal", "Mandal"))
    levels.append(("district", "District"))
    
    all_alerts = []
    
    for region_col, level_name in levels:
        print_header(f"ADAPTIVE ISOLATION FOREST — {level_name.upper()} LEVEL")
        
        daily = build_daily_series(merged, region_col)
        if daily.empty:
            print(f"  No data for {level_name} level.")
            continue
        
        # Build features (with adaptive contamination per pair)
        print_subheader("Building enhanced features")
        features = build_features(daily, region_col)
        if features.empty:
            print(f"  Insufficient data at {level_name} level.")
            continue
        print(f"  Feature matrix: {len(features):,} rows × {len(FEATURE_COLS)} features")
        
        # Run Isolation Forest
        print_subheader("Running Adaptive Isolation Forest")
        results = detect_anomalies(features, region_col)
        
        if results.empty:
            continue
        
        results["severity"] = results.apply(classify_severity, axis=1)
        results["level"] = level_name
        
        anomalies = results[results["anomaly_label"] == -1].copy()
        
        if anomalies.empty:
            print(f"\n  ✅ No anomalies at {level_name} level.")
            continue
        
        # Print top anomalies
        print_subheader(f"Top Anomalies — {level_name}")
        display = (
            anomalies
            .sort_values("anomaly_score")
            .head(30)
            [[
                "severity", region_col, "complaint_name", "date",
                "case_count", "rolling_7d_mean", "ratio_to_mean",
                "anomaly_score", "_contamination"
            ]]
            .copy()
        )
        display["rolling_7d_mean"] = display["rolling_7d_mean"].round(1)
        display["ratio_to_mean"] = display["ratio_to_mean"].round(1)
        display["anomaly_score"] = display["anomaly_score"].round(3)
        display["_contamination"] = display["_contamination"].round(3)
        display.columns = ["Severity", level_name, "Disease", "Date",
                          "Cases", "7d_Mean", "Spike", "Score", "Contam%"]
        print(f"\n{display.to_string(index=False)}")
        
        all_alerts.append(anomalies)
    
    # ── Save report ──────────────────────────────────────────────────────
    print_header("SAVING REPORT")
    if all_alerts:
        report = pd.concat(all_alerts, ignore_index=True)
        # Clean up internal columns
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
  
  Improvements over base Isolation Forest:
    ✅ Adaptive contamination (dynamic per disease-region pair)
    ✅ 9 features vs 6 (rolling_14d, lag_7d, cumulative_3d)
    ✅ Higher MIN_TOTAL_CASES (20 vs 10) to reduce noise
""")
if __name__ == "__main__":
    main()
