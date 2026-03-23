"""
Disease Outbreak Detection using Facebook Prophet
===================================================
Uses Prophet time series forecasting to detect outbreaks by comparing
actual case counts against predicted upper confidence bounds.
How it works:
  - For each (disease, region), Prophet learns:
      * Trend:      overall direction of case counts
      * Weekly:     day-of-week seasonality (e.g., fewer reports on weekends)
      * Residuals:  random noise
  - Prophet generates a forecast with confidence intervals (default 95%)
  - If actual cases > upper bound (yhat_upper) → OUTBREAK ALERT
  - The further above the bound, the more severe the alert
Usage:
  1. Place ip.csv and pincode_directory.csv in this directory
  2. pip install prophet pandas numpy
  3. python outbreak_prophet.py
Key parameters:
  - CONFIDENCE_INTERVAL: 0.95 (95%) — higher = fewer false alerts
  - CRITICAL_MULTIPLIER: 1.5 — if actual > 1.5× upper bound → CRITICAL
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
import os
import sys
import logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
# Import shared utilities
from data_utils import (
    load_ip_dataset, load_pincode_mapping, merge_geography,
    build_daily_series, print_header, print_subheader, SCRIPT_DIR
)
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Prophet parameters
CONFIDENCE_INTERVAL = 0.95    # Confidence interval width (95%)
CRITICAL_MULTIPLIER = 1.5     # actual > 1.5 × yhat_upper → CRITICAL
# Filtering
MIN_TOTAL_CASES = 10          # Skip disease-region pairs with very few cases
MIN_DAYS = 21                 # Need at least 21 days for Prophet to learn patterns
# Output
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "outbreak_prophet_report.csv")
# ─────────────────────────────────────────────────────────────────────────────
# PROPHET OUTBREAK DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies_prophet(daily_df, region_col):
    """
    Run Prophet for each (disease, region) pair.
    
    Prophet expects two columns:
      - ds: date (datetime)
      - y:  value (case count)
    
    It fits a model that captures trend + weekly seasonality,
    then generates yhat_upper (95% confidence). Days where
    actual > yhat_upper are flagged as anomalies.
    """
    all_alerts = []
    total_pairs = 0
    skipped = 0
    error_count = 0
    
    groups = list(daily_df.groupby([region_col, "complaint_name"]))
    total_groups = len(groups)
    
    for idx, ((region, disease), group) in enumerate(groups):
        total_pairs += 1
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing {idx+1}/{total_groups}: {disease} in {region}...", end="\r")
        
        group = group.sort_values("date").copy()
        total_cases = group["case_count"].sum()
        
        if total_cases < MIN_TOTAL_CASES or len(group) < MIN_DAYS:
            skipped += 1
            continue
        
        # Prepare Prophet input
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(group["date"]),
            "y": group["case_count"].values.astype(float)
        })
        
        try:
            # Fit Prophet model
            model = Prophet(
                interval_width=CONFIDENCE_INTERVAL,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Only 6 months of data
                changepoint_prior_scale=0.1,  # Moderate flexibility for trend changes
            )
            model.fit(prophet_df)
            
            # In-sample prediction (detect anomalies in existing data)
            forecast = model.predict(prophet_df)
            
            # Merge actual values with forecast
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            result["actual"] = prophet_df["y"].values
            result["date"] = group["date"].values
            result[region_col] = region
            result["complaint_name"] = disease
            
            # Calculate how far above the upper bound
            result["excess"] = result["actual"] - result["yhat_upper"]
            result["excess_ratio"] = np.where(
                result["yhat_upper"] > 0,
                result["actual"] / result["yhat_upper"],
                result["actual"]
            )
            
            # Flag anomalies: actual > yhat_upper
            anomalies = result[result["actual"] > result["yhat_upper"]].copy()
            
            if len(anomalies) > 0:
                # Classify severity
                anomalies["severity"] = np.where(
                    anomalies["excess_ratio"] >= CRITICAL_MULTIPLIER,
                    "🔴 CRITICAL",
                    "⚠️ ALERT"
                )
                all_alerts.append(anomalies)
        
        except Exception as e:
            error_count += 1
            continue
    
    print(f"\n  Analyzed: {total_pairs} pairs | Skipped: {skipped} (insufficient data) | Errors: {error_count}")
    
    if all_alerts:
        return pd.concat(all_alerts, ignore_index=True)
    return pd.DataFrame()
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print_header("OUTBREAK DETECTION — FACEBOOK PROPHET")
    print(f"  Confidence interval: {CONFIDENCE_INTERVAL*100:.0f}%")
    print(f"  Critical multiplier: {CRITICAL_MULTIPLIER}× upper bound")
    print(f"  Min cases threshold: {MIN_TOTAL_CASES}")
    print(f"  Min days required: {MIN_DAYS}")
    
    # ── Load data ────────────────────────────────────────────────────────
    ip_df = load_ip_dataset()
    pin_mapping, has_mandal = load_pincode_mapping()
    merged = merge_geography(ip_df, pin_mapping, has_mandal)
    
    # ── Run at each geographic level ─────────────────────────────────────
    levels = [("district", "District")]  # Start with district (fastest)
    if has_mandal:
        levels.append(("mandal", "Mandal"))
    levels.append(("pincode", "Pincode"))
    
    all_alerts = []
    
    for region_col, level_name in levels:
        print_header(f"PROPHET ANALYSIS — {level_name.upper()} LEVEL")
        
        # Build daily time series
        daily = build_daily_series(merged, region_col)
        if daily.empty:
            print(f"  No data for {level_name} level.")
            continue
        
        unique_pairs = daily.groupby([region_col, "complaint_name"]).ngroups
        print(f"  Disease-region pairs to analyze: {unique_pairs}")
        
        if level_name == "Pincode" and unique_pairs > 500:
            print(f"  ⚠️ Large number of pairs ({unique_pairs}). This may take a while...")
            print(f"     Tip: For faster results, run district-level only.")
        
        # Run Prophet
        print_subheader("Running Prophet forecasting")
        anomalies = detect_anomalies_prophet(daily, region_col)
        
        if anomalies.empty:
            print(f"\n  ✅ No outbreaks detected at {level_name} level.")
            continue
        
        # Print top anomalies
        print_subheader(f"Top Anomalies at {level_name} Level")
        display = (
            anomalies
            .sort_values("excess", ascending=False)
            .head(30)
            [[
                "severity", region_col, "complaint_name", "date",
                "actual", "yhat", "yhat_upper", "excess"
            ]]
            .copy()
        )
        display["yhat"] = display["yhat"].round(1)
        display["yhat_upper"] = display["yhat_upper"].round(1)
        display["excess"] = display["excess"].round(1)
        display.columns = ["Severity", level_name, "Disease", "Date",
                          "Actual", "Predicted", "Upper_95%", "Excess"]
        
        print(f"\n{display.to_string(index=False)}")
        
        anomalies["level"] = level_name
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
  Total anomalies:   {total}
  🔴 CRITICAL:       {critical}
  ⚠️ ALERT:          {alert}
  
  Output: {OUTPUT_PATH}
  
  How Prophet detects outbreaks:
    1. Learns trend + weekly seasonality from your 6 months of data
    2. Builds a {CONFIDENCE_INTERVAL*100:.0f}% confidence interval [yhat_lower, yhat_upper]
    3. Days where actual cases EXCEED yhat_upper → flagged as anomaly
    4. If actual > {CRITICAL_MULTIPLIER}× yhat_upper → CRITICAL alert
  
  Tip: Increase CONFIDENCE_INTERVAL (currently {CONFIDENCE_INTERVAL}) for fewer
       false positives, decrease for more sensitivity.
""")
if __name__ == "__main__":
    main()
