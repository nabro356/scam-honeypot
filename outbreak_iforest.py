"""
Disease Outbreak Detection using Isolation Forest
===================================================
Uses unsupervised Isolation Forest algorithm to detect anomalous disease
spikes at pincode, mandal, and district levels.
How it works:
  - For each (disease, region), we build features from the daily case time series:
      * case_count:       today's cases
      * rolling_7d_mean:  average cases over past 7 days
      * rolling_7d_std:   std dev over past 7 days
      * ratio_to_mean:    today / 7-day mean (spike ratio)
      * day_of_week:      0=Mon, 6=Sun (captures weekly patterns)
      * diff_1d:          change from yesterday
  - Isolation Forest identifies points that are "easy to isolate" (outliers)
    by randomly partitioning the feature space
  - Points with anomaly_score = -1 are flagged as outbreaks
Usage:
  1. Place ip.csv and pincode_directory.csv in this directory
  2. pip install scikit-learn pandas numpy
  3. python outbreak_iforest.py
Contamination parameter (default 0.05 = top 5%) controls sensitivity.
Lower = fewer alerts, Higher = more alerts.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import os
import sys
warnings.filterwarnings("ignore")
# Import shared utilities
from data_utils import (
    load_ip_dataset, load_pincode_mapping, merge_geography,
    build_daily_series, print_header, print_subheader, SCRIPT_DIR
)
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest parameters
CONTAMINATION = 0.05          # Expected proportion of anomalies (5%)
N_ESTIMATORS = 200            # Number of trees in the forest
RANDOM_STATE = 42             # Reproducibility
# Feature engineering parameters
ROLLING_WINDOW = 7            # Days for rolling features
MIN_TOTAL_CASES = 10          # Skip (disease, region) pairs with very few total cases
MIN_DAYS = 14                 # Minimum days of data needed
# Output
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "outbreak_iforest_report.csv")
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def build_features(daily_df, region_col):
    """
    Build features for Isolation Forest from daily case count time series.
    
    Features per day:
      - case_count:      raw count
      - rolling_7d_mean: 7-day rolling average (past days, excl. today)
      - rolling_7d_std:  7-day rolling std
      - ratio_to_mean:   case_count / rolling_7d_mean (spike multiplier)
      - diff_1d:         day-over-day change in cases
      - day_of_week:     0-6 (captures weekly reporting patterns)
    """
    all_features = []
    
    for (region, disease), group in daily_df.groupby([region_col, "complaint_name"]):
        group = group.sort_values("date").copy()
        
        total_cases = group["case_count"].sum()
        if total_cases < MIN_TOTAL_CASES or len(group) < MIN_DAYS:
            continue
        
        # Rolling statistics (using past data only — shift(1))
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
        
        # Spike ratio: how many times the mean is today's count?
        group["ratio_to_mean"] = np.where(
            group["rolling_7d_mean"] > 0,
            group["case_count"] / group["rolling_7d_mean"],
            group["case_count"]  # If mean is 0, use raw count as ratio
        )
        
        # Day-over-day change
        group["diff_1d"] = group["case_count"].diff()
        
        # Day of week
        group["day_of_week"] = pd.to_datetime(group["date"]).dt.dayofweek
        
        # Drop rows where rolling stats aren't available yet
        group = group.dropna(subset=["rolling_7d_mean", "rolling_7d_std", "diff_1d"])
        
        if len(group) < MIN_DAYS:
            continue
        
        all_features.append(group)
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# ISOLATION FOREST DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies_iforest(features_df, region_col):
    """
    Run Isolation Forest per (disease, region) pair.
    Returns DataFrame with anomaly labels and scores.
    """
    feature_cols = ["case_count", "rolling_7d_mean", "rolling_7d_std",
                    "ratio_to_mean", "diff_1d", "day_of_week"]
    
    all_results = []
    anomaly_count = 0
    total_pairs = 0
    
    for (region, disease), group in features_df.groupby([region_col, "complaint_name"]):
        total_pairs += 1
        group = group.copy()
        
        X = group[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        group["anomaly_label"] = model.fit_predict(X_scaled)  # 1=normal, -1=anomaly
        group["anomaly_score"] = model.decision_function(X_scaled)  # Lower = more anomalous
        
        # Filter anomalies (-1)
        anomalies = group[group["anomaly_label"] == -1]
        anomaly_count += len(anomalies)
        
        all_results.append(group)
    
    print(f"  Analyzed {total_pairs} (disease, region) pairs")
    print(f"  Total anomalies detected: {anomaly_count}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
def classify_severity(row):
    """Classify anomaly severity based on anomaly score and spike ratio."""
    if row["anomaly_label"] != -1:
        return "✅ Normal"
    if row["anomaly_score"] < -0.3 or row["ratio_to_mean"] > 5:
        return "🔴 CRITICAL"
    return "⚠️ ALERT"
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print_header("OUTBREAK DETECTION — ISOLATION FOREST")
    print(f"  Contamination: {CONTAMINATION} ({CONTAMINATION*100:.0f}% expected anomalies)")
    print(f"  Trees: {N_ESTIMATORS}")
    print(f"  Rolling window: {ROLLING_WINDOW} days")
    print(f"  Min cases threshold: {MIN_TOTAL_CASES}")
    
    # ── Load data ────────────────────────────────────────────────────────
    ip_df = load_ip_dataset()
    pin_mapping, has_mandal = load_pincode_mapping()
    merged = merge_geography(ip_df, pin_mapping, has_mandal)
    
    # ── Run at each geographic level ─────────────────────────────────────
    levels = [("pincode", "Pincode")]
    if has_mandal:
        levels.append(("mandal", "Mandal"))
    levels.append(("district", "District"))
    
    all_alerts = []
    
    for region_col, level_name in levels:
        print_header(f"ISOLATION FOREST — {level_name.upper()} LEVEL")
        
        # Build daily time series
        daily = build_daily_series(merged, region_col)
        if daily.empty:
            print(f"  No data for {level_name} level.")
            continue
        
        # Build features
        print_subheader("Building features")
        features = build_features(daily, region_col)
        if features.empty:
            print(f"  Insufficient data for feature engineering at {level_name} level.")
            continue
        print(f"  Feature matrix: {len(features):,} rows × 6 features")
        
        # Run Isolation Forest
        print_subheader("Running Isolation Forest")
        results = detect_anomalies_iforest(features, region_col)
        
        if results.empty:
            continue
        
        # Classify severity
        results["severity"] = results.apply(classify_severity, axis=1)
        results["level"] = level_name
        
        # Filter to anomalies only
        anomalies = results[results["anomaly_label"] == -1].copy()
        
        if anomalies.empty:
            print(f"\n  ✅ No anomalies detected at {level_name} level.")
            continue
        
        # Print top anomalies
        print_subheader(f"Top Anomalies at {level_name} Level")
        display = (
            anomalies
            .sort_values("anomaly_score")
            .head(30)
            [[
                "severity", region_col, "complaint_name", "date",
                "case_count", "rolling_7d_mean", "ratio_to_mean", "anomaly_score"
            ]]
            .copy()
        )
        display["rolling_7d_mean"] = display["rolling_7d_mean"].round(1)
        display["ratio_to_mean"] = display["ratio_to_mean"].round(1)
        display["anomaly_score"] = display["anomaly_score"].round(3)
        display.columns = ["Severity", level_name, "Disease", "Date",
                          "Cases", "7d_Mean", "Spike_Ratio", "Anomaly_Score"]
        
        print(f"\n{display.to_string(index=False)}")
        
        all_alerts.append(anomalies)
    
    # ── Save report ──────────────────────────────────────────────────────
    print_header("SAVING REPORT")
    if all_alerts:
        report = pd.concat(all_alerts, ignore_index=True)
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
  
  Tip: Adjust CONTAMINATION (currently {CONTAMINATION}) to control sensitivity.
       Lower = fewer alerts, Higher = more alerts.
""")
if __name__ == "__main__":
    main()
