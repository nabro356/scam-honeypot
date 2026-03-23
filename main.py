"""
Disease Outbreak Detection System
==================================
Detects disease outbreaks from health complaint data using:
  1. SNOMED CT complaint directory mapping
  2. Pincode → Mandal/District mapping (India Post directory)
  3. Z-score anomaly detection at pincode, mandal, and district levels
  4. Multi-region escalation alerts
Usage:
  1. Place your dataset as 'ip.csv' in the same directory as this script.
  2. Place the India Post pincode directory as 'pincode_directory.csv' in the same directory.
     - Download from: https://data.gov.in/resource/all-india-pincode-directory
     - Or from Kaggle: search "All India Pincode Directory"
     - Expected columns: CircleName, RegionName, DivisionName, OfficeName,
       Pincode, OfficeType, Delivery, District, StateName, Latitude, Longitude
  3. (Optional) Place a 'pincode_mandal_mapping.csv' with columns: pincode, mandal
     for exact mandal-level mapping. Without it, DivisionName is used as proxy.
  4. Run: python outbreak_detection.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import sys
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# File paths (place files in same directory as this script)
IP_DATASET_PATH = os.path.join(SCRIPT_DIR, "ip.csv")
PINCODE_DIR_PATH = os.path.join(SCRIPT_DIR, "pincode_directory.csv")
# State filter
TARGET_STATE = "ANDHRA PRADESH"
# Z-score anomaly detection parameters
ROLLING_WINDOW_DAYS = 14       # Rolling window for baseline computation
Z_ALERT_THRESHOLD = 2.0        # Z > 2 → ALERT (top 2.3%, ~1 in 44 days by chance)
Z_CRITICAL_THRESHOLD = 3.0     # Z > 3 → CRITICAL (top 0.13%, ~1 in 740 days by chance)
MIN_CASES_FOR_ALERT = 3        # Minimum daily cases to trigger any alert (noise filter)
# Escalation: if >= N mandals in same district flag same disease within WINDOW days
ESCALATION_MANDAL_COUNT = 2
ESCALATION_WINDOW_DAYS = 7
# Output files
COMPLAINT_DIR_OUTPUT = os.path.join(SCRIPT_DIR, "complaint_directory.csv")
OUTBREAK_REPORT_OUTPUT = os.path.join(SCRIPT_DIR, "outbreak_report.csv")
# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def print_header(title):
    """Print a formatted section header."""
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")
def load_ip_dataset(path):
    """Load and clean the health complaint (IP) dataset."""
    print(f"\nLoading IP dataset from: {path}")
    
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        print("Please place your dataset as 'ip.csv' in the same directory as this script.")
        sys.exit(1)
    
    df = pd.read_csv(path)
    print(f"  Raw records: {len(df):,}")
    
    # Standardize column names (handle variations)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if "health" in cl and "id" in cl:
            col_map[col] = "health_id"
        elif cl in ("complaint", "complaint_code", "snomed_code", "snomed_ct_code"):
            col_map[col] = "complaint"
        elif cl in ("complaint_name", "complaint_desc", "snomed_name", "ct_name"):
            col_map[col] = "complaint_name"
        elif cl in ("pincode", "pin_code", "zip", "postal_code"):
            col_map[col] = "pincode"
        elif cl in ("timestamp", "time_stamp", "unix_timestamp", "created_at"):
            col_map[col] = "timestamp"
    
    df = df.rename(columns=col_map)
    
    # Validate required columns
    required = ["health_id", "complaint", "complaint_name", "pincode", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        print("  Please ensure your CSV has: health_id, complaint, complaint_name, pincode, timestamp")
        sys.exit(1)
    
    # Convert timestamp to datetime and extract date
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df["date"] = df["datetime"].dt.date
    
    # Clean pincode
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
    
    # Report per-field missing/invalid counts before dropping
    missing_pincode = df["pincode"].isna().sum()
    missing_date = df["date"].isna().sum()
    missing_complaint = df["complaint_name"].isna().sum() + (df["complaint_name"].str.strip() == "").sum()
    print(f"\n  Missing/invalid values:")
    print(f"    pincode:        {missing_pincode:,}")
    print(f"    timestamp/date: {missing_date:,}")
    print(f"    complaint_name: {missing_complaint:,}")
    
    df = df.dropna(subset=["pincode", "date", "complaint_name"])
    df = df[df["complaint_name"].str.strip() != ""]
    df["pincode"] = df["pincode"].astype(int)
    
    # Clean complaint names
    df["complaint_name"] = df["complaint_name"].str.strip().str.title()
    df["complaint"] = df["complaint"].astype(str).str.strip()
    
    print(f"\n  Clean records: {len(df):,} (dropped {len(pd.read_csv(path)) - len(df):,})")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique complaints: {df['complaint_name'].nunique()}")
    print(f"  Unique pincodes: {df['pincode'].nunique()}")
    
    return df
def build_complaint_directory(df):
    """Build and save SNOMED CT complaint directory mapping."""
    print_header("STEP 1: SNOMED CT COMPLAINT DIRECTORY")
    
    directory = (
        df.groupby(["complaint", "complaint_name"])
        .agg(total_cases=("health_id", "count"))
        .reset_index()
        .sort_values("total_cases", ascending=False)
    )
    
    directory.columns = ["SNOMED_CT_Code", "Complaint_Name", "Total_Cases"]
    
    print(f"\nFound {len(directory)} unique complaint mappings:\n")
    print(directory.to_string(index=False))
    
    # Save to CSV
    directory.to_csv(COMPLAINT_DIR_OUTPUT, index=False)
    print(f"\nSaved complaint directory to: {COMPLAINT_DIR_OUTPUT}")
    
    return directory
def load_pincode_directory(path):
    """
    Load India Post pincode directory and filter for target state.
    
    Expected columns from India Post CSV:
      CircleName, RegionName, DivisionName, OfficeName, Pincode,
      OfficeType, Delivery, District, StateName, Latitude, Longitude
    
    Since India Post CSV does NOT have a Taluk/Mandal column, we use
    DivisionName as a mandal-level proxy. Postal divisions in AP provide
    a reasonable sub-district geographic grouping.
    
    If you have a separate pincode-to-mandal mapping CSV, set
    MANDAL_MAPPING_PATH at the top of this script.
    """
    print_header("STEP 2: PINCODE → MANDAL/DISTRICT MAPPING")
    
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        print("Please download the India Post pincode directory:")
        print("  - https://data.gov.in/resource/all-india-pincode-directory")
        print("  - Or search Kaggle for 'All India Pincode Directory'")
        print(f"  - Save as: {path}")
        sys.exit(1)
    
    print(f"\nLoading India Post pincode directory from: {path}")
    
    # Try different encodings
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            pin_df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        print("ERROR: Could not read pincode directory file.")
        sys.exit(1)
    
    print(f"  Total records in directory: {len(pin_df):,}")
    print(f"  Columns: {list(pin_df.columns)}")
    
    # ── Standardize column names ──────────────────────────────────────
    # Handles both the India Post format and other common formats
    col_map = {}
    for col in pin_df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("pincode", "pin_code", "pin"):
            col_map[col] = "pincode"
        elif cl in ("taluk", "taluka", "taluka_name", "taluk_name", "mandal", "tehsil"):
            col_map[col] = "mandal"  # Direct mandal column (if present)
        elif cl in ("divisionname", "division_name", "division"):
            col_map[col] = "division"  # Fallback for mandal-level grouping
        elif cl in ("district", "district_name", "districtname"):
            col_map[col] = "district"
        elif cl in ("statename", "state_name", "state"):
            col_map[col] = "state"
        elif cl in ("regionname", "region_name", "region"):
            col_map[col] = "region"
        elif cl in ("officename", "office_name"):
            col_map[col] = "office_name"
    
    pin_df = pin_df.rename(columns=col_map)
    
    # Check required columns
    required = ["pincode", "district", "state"]
    missing = [c for c in required if c not in pin_df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"  Available: {list(pin_df.columns)}")
        print("  Please check your pincode directory file format.")
        sys.exit(1)
    
    # ── Determine mandal-level column ─────────────────────────────────
    # Priority: mandal (Taluk) > division (DivisionName) > none
    has_mandal = False
    if "mandal" in pin_df.columns:
        has_mandal = True
        print("  ✅ Found direct Taluk/Mandal column.")
    elif "division" in pin_df.columns:
        # Use DivisionName as mandal-level proxy
        pin_df["mandal"] = pin_df["division"]
        has_mandal = True
        print("  ℹ️  No Taluk/Mandal column found.")
        print("      Using 'DivisionName' as mandal-level grouping (postal divisions).")
        print("      For exact mandal mapping, provide a separate CSV with pincode→mandal.")
    else:
        print("  ⚠️  No mandal-level column found. District-level analysis only.")
    
    # ── Check for separate mandal mapping ─────────────────────────────
    mandal_csv = os.path.join(SCRIPT_DIR, "pincode_mandal_mapping.csv")
    if os.path.exists(mandal_csv):
        print(f"\n  Found separate mandal mapping: {mandal_csv}")
        mandal_map = pd.read_csv(mandal_csv)
        # Expect columns: pincode, mandal
        mandal_map.columns = [c.strip().lower().replace(" ", "_") for c in mandal_map.columns]
        if "pincode" in mandal_map.columns and "mandal" in mandal_map.columns:
            mandal_map["pincode"] = pd.to_numeric(mandal_map["pincode"], errors="coerce")
            mandal_map["mandal"] = mandal_map["mandal"].astype(str).str.strip().str.title()
            # Merge mandal info into pincode directory
            pin_df = pin_df.drop(columns=["mandal"], errors="ignore")
            pin_df = pin_df.merge(mandal_map[["pincode", "mandal"]], on="pincode", how="left")
            has_mandal = True
            print("  ✅ Loaded mandal mapping successfully.")
    
    # ── Filter for target state ───────────────────────────────────────
    pin_df["state"] = pin_df["state"].astype(str).str.strip().str.upper()
    
    # Try different state name variations
    state_variations = [
        TARGET_STATE,
        TARGET_STATE.replace(" ", ""),
        "AP",
        "A.P.",
        "ANDHRA  PRADESH",   # double space
        "ANDHRA PRADESH ",   # trailing space
    ]
    
    state_mask = pin_df["state"].isin(state_variations)
    
    if state_mask.sum() == 0:
        # Show available states for debugging
        print(f"\n  WARNING: No records found for '{TARGET_STATE}'")
        available = sorted(pin_df["state"].unique())
        print(f"  Available states ({len(available)}): {available[:15]}")
        print("  Proceeding with all pincodes (will match on join)...")
        ap_df = pin_df.copy()
    else:
        ap_df = pin_df[state_mask].copy()
        print(f"  Records for {TARGET_STATE}: {len(ap_df):,}")
    
    # ── Clean and deduplicate ─────────────────────────────────────────
    ap_df["pincode"] = pd.to_numeric(ap_df["pincode"], errors="coerce")
    ap_df = ap_df.dropna(subset=["pincode"])
    ap_df["pincode"] = ap_df["pincode"].astype(int)
    ap_df["district"] = ap_df["district"].astype(str).str.strip().str.title()
    if has_mandal:
        ap_df["mandal"] = ap_df["mandal"].astype(str).str.strip().str.title()
        # Clean up 'Nan' strings from fillna
        ap_df.loc[ap_df["mandal"].isin(["Nan", "None", ""]), "mandal"] = "Unknown"
    
    # Get unique pincode → mandal/district mapping
    group_cols = ["pincode", "district"]
    if has_mandal:
        group_cols = ["pincode", "mandal", "district"]
    
    mapping = ap_df[group_cols].drop_duplicates(subset=["pincode"]).reset_index(drop=True)
    
    print(f"\n  Unique pincode mappings: {len(mapping):,}")
    print(f"  Districts: {mapping['district'].nunique()}")
    if has_mandal:
        mandal_count = mapping[mapping["mandal"] != "Unknown"]["mandal"].nunique()
        print(f"  Mandals/Divisions: {mandal_count}")
    
    return mapping, has_mandal
def merge_with_geography(ip_df, pin_mapping, has_mandal):
    """Merge IP dataset with geographic mapping."""
    print_subheader("Merging IP data with geographic mapping")
    
    merged = ip_df.merge(pin_mapping, on="pincode", how="left")
    
    # Report coverage
    total = len(merged)
    mapped = merged["district"].notna().sum()
    unmapped = total - mapped
    
    print(f"  Total records: {total:,}")
    print(f"  Mapped to district: {mapped:,} ({100*mapped/total:.1f}%)")
    print(f"  Unmapped: {unmapped:,} ({100*unmapped/total:.1f}%)")
    
    if unmapped > 0:
        unmapped_pins = merged.loc[merged["district"].isna(), "pincode"].unique()
        print(f"  Unmapped pincodes ({len(unmapped_pins)}): {sorted(unmapped_pins)[:20]}")
        if len(unmapped_pins) > 20:
            print(f"    ... and {len(unmapped_pins) - 20} more")
    
    # Fill unmapped with "Unknown"
    merged["district"] = merged["district"].fillna("Unknown")
    if has_mandal:
        merged["mandal"] = merged["mandal"].fillna("Unknown")
    
    return merged
def aggregate_cases(df, group_cols, level_name):
    """Aggregate case counts by given grouping columns + date."""
    agg = (
        df.groupby(group_cols + ["complaint_name", "date"])
        .agg(case_count=("health_id", "count"))
        .reset_index()
        .sort_values(group_cols + ["complaint_name", "date"])
    )
    return agg
def compute_zscore_alerts(agg_df, region_col, level_name):
    """
    Compute Z-score based anomaly detection for each (disease, region) pair.
    
    Z-score = (today's count - rolling_mean) / rolling_std
    
    Thresholds (from CDC EARS methodology):
      Z > 2.0 → ALERT   (value in top 2.3%, ~1 in 44 days by chance)
      Z > 3.0 → CRITICAL (value in top 0.13%, ~1 in 740 days by chance)
    """
    alerts = []
    
    for (disease, region), group in agg_df.groupby(["complaint_name", region_col]):
        group = group.sort_values("date").copy()
        
        # Need at least some data for rolling stats
        if len(group) < ROLLING_WINDOW_DAYS:
            continue
        
        # Ensure continuous date range (fill missing dates with 0 cases)
        date_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        full_dates = pd.DataFrame({"date": date_range.date})
        group = full_dates.merge(group, on="date", how="left")
        group["case_count"] = group["case_count"].fillna(0)
        group["complaint_name"] = disease
        group[region_col] = region
        
        # Compute rolling statistics
        group["rolling_mean"] = (
            group["case_count"]
            .rolling(window=ROLLING_WINDOW_DAYS, min_periods=max(7, ROLLING_WINDOW_DAYS // 2))
            .mean()
            .shift(1)  # Don't include today in the baseline
        )
        group["rolling_std"] = (
            group["case_count"]
            .rolling(window=ROLLING_WINDOW_DAYS, min_periods=max(7, ROLLING_WINDOW_DAYS // 2))
            .std()
            .shift(1)
        )
        
        # Compute Z-score (guard against zero std)
        group["z_score"] = np.where(
            group["rolling_std"] > 0,
            (group["case_count"] - group["rolling_mean"]) / group["rolling_std"],
            0
        )
        
        # Flag alerts
        alert_mask = (
            (group["z_score"] >= Z_ALERT_THRESHOLD) &
            (group["case_count"] >= MIN_CASES_FOR_ALERT) &
            (group["rolling_mean"].notna())
        )
        
        flagged = group[alert_mask].copy()
        
        if len(flagged) > 0:
            flagged["level"] = level_name
            flagged["severity"] = np.where(
                flagged["z_score"] >= Z_CRITICAL_THRESHOLD, "🔴 CRITICAL", "⚠️ ALERT"
            )
            alerts.append(flagged[["level", region_col, "complaint_name", "date",
                                   "case_count", "rolling_mean", "rolling_std",
                                   "z_score", "severity"]])
    
    if alerts:
        return pd.concat(alerts, ignore_index=True)
    else:
        return pd.DataFrame()
def detect_escalations(mandal_alerts, merged_df, has_mandal):
    """
    Detect district-level escalations: if >=N mandals in the same district
    flag the same disease within a time window → DISTRICT OUTBREAK.
    """
    if not has_mandal or mandal_alerts.empty:
        return pd.DataFrame()
    
    # Get mandal → district mapping
    mandal_to_district = (
        merged_df[["mandal", "district"]]
        .drop_duplicates()
        .set_index("mandal")["district"]
        .to_dict()
    )
    
    mandal_alerts = mandal_alerts.copy()
    mandal_alerts["district"] = mandal_alerts["mandal"].map(mandal_to_district)
    mandal_alerts["date"] = pd.to_datetime(mandal_alerts["date"])
    
    escalations = []
    
    for (disease, district), group in mandal_alerts.groupby(["complaint_name", "district"]):
        if district == "Unknown":
            continue
        
        # Sort by date
        group = group.sort_values("date")
        
        # Sliding window: check if >=N mandals flagged within WINDOW days
        for _, row in group.iterrows():
            window_start = row["date"] - timedelta(days=ESCALATION_WINDOW_DAYS)
            window_end = row["date"]
            
            window_alerts = group[
                (group["date"] >= window_start) & (group["date"] <= window_end)
            ]
            
            unique_mandals = window_alerts["mandal"].nunique()
            
            if unique_mandals >= ESCALATION_MANDAL_COUNT:
                escalations.append({
                    "disease": disease,
                    "district": district,
                    "date": row["date"].date(),
                    "mandals_affected": unique_mandals,
                    "mandal_names": ", ".join(sorted(window_alerts["mandal"].unique())),
                    "total_cases_in_window": int(window_alerts["case_count"].sum()),
                    "max_z_score": round(window_alerts["z_score"].max(), 2),
                    "severity": "🚨 DISTRICT OUTBREAK"
                })
    
    if escalations:
        esc_df = pd.DataFrame(escalations).drop_duplicates(
            subset=["disease", "district", "date"]
        )
        return esc_df
    
    return pd.DataFrame()
def print_aggregation_summary(agg_df, region_col, level_name, top_n=20):
    """Print top disease counts for a given aggregation level."""
    print_subheader(f"Top {top_n} Disease Counts by {level_name}")
    
    summary = (
        agg_df.groupby([region_col, "complaint_name"])
        .agg(
            total_cases=("case_count", "sum"),
            days_reported=("date", "nunique"),
            first_seen=("date", "min"),
            last_seen=("date", "max")
        )
        .reset_index()
        .sort_values("total_cases", ascending=False)
        .head(top_n)
    )
    
    print(f"\n{summary.to_string(index=False)}")
def print_alerts(alerts_df, level_name):
    """Print outbreak alerts for a given level."""
    if alerts_df.empty:
        print(f"\n  ✅ No anomalies detected at {level_name} level.")
        return
    
    print(f"\n  Found {len(alerts_df)} alert(s) at {level_name} level:\n")
    
    # Sort by z-score descending
    display = alerts_df.sort_values("z_score", ascending=False).copy()
    display["rolling_mean"] = display["rolling_mean"].round(1)
    display["rolling_std"] = display["rolling_std"].round(1)
    display["z_score"] = display["z_score"].round(2)
    
    region_col = [c for c in display.columns if c not in [
        "level", "complaint_name", "date", "case_count",
        "rolling_mean", "rolling_std", "z_score", "severity"
    ]][0]
    
    cols = ["severity", region_col, "complaint_name", "date",
            "case_count", "rolling_mean", "z_score"]
    
    print(display[cols].to_string(index=False))
# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print_header("DISEASE OUTBREAK DETECTION SYSTEM")
    print(f"  Target State: {TARGET_STATE}")
    print(f"  Z-Score Thresholds: ALERT > {Z_ALERT_THRESHOLD}, CRITICAL > {Z_CRITICAL_THRESHOLD}")
    print(f"  Rolling Window: {ROLLING_WINDOW_DAYS} days")
    print(f"  Min Cases for Alert: {MIN_CASES_FOR_ALERT}")
    print(f"  Escalation: >= {ESCALATION_MANDAL_COUNT} mandals within {ESCALATION_WINDOW_DAYS} days")
    
    # ── Step 0: Load data ────────────────────────────────────────────────
    ip_df = load_ip_dataset(IP_DATASET_PATH)
    
    # ── Step 1: Complaint directory ──────────────────────────────────────
    complaint_dir = build_complaint_directory(ip_df)
    
    # ── Step 2: Pincode mapping ──────────────────────────────────────────
    pin_mapping, has_mandal = load_pincode_directory(PINCODE_DIR_PATH)
    
    # ── Step 3: Merge ────────────────────────────────────────────────────
    merged = merge_with_geography(ip_df, pin_mapping, has_mandal)
    
    # ── Step 4: Aggregate at all levels ──────────────────────────────────
    print_header("STEP 3: DISEASE AGGREGATION BY REGION")
    
    # Pincode level
    pincode_agg = aggregate_cases(merged, ["pincode"], "Pincode")
    print_aggregation_summary(pincode_agg, "pincode", "Pincode")
    
    # Mandal level
    if has_mandal:
        mandal_agg = aggregate_cases(merged, ["mandal"], "Mandal")
        print_aggregation_summary(mandal_agg, "mandal", "Mandal")
    
    # District level
    district_agg = aggregate_cases(merged, ["district"], "District")
    print_aggregation_summary(district_agg, "district", "District")
    
    # ── Step 5: Z-Score anomaly detection ────────────────────────────────
    print_header("STEP 4: OUTBREAK DETECTION (Z-SCORE ANOMALY)")
    
    print_subheader("Pincode-Level Alerts")
    pincode_alerts = compute_zscore_alerts(pincode_agg, "pincode", "Pincode")
    print_alerts(pincode_alerts, "Pincode")
    
    if has_mandal:
        print_subheader("Mandal-Level Alerts")
        mandal_alerts = compute_zscore_alerts(mandal_agg, "mandal", "Mandal")
        print_alerts(mandal_alerts, "Mandal")
    else:
        mandal_alerts = pd.DataFrame()
    
    print_subheader("District-Level Alerts")
    district_alerts = compute_zscore_alerts(district_agg, "district", "District")
    print_alerts(district_alerts, "District")
    
    # ── Step 6: Multi-region escalation ──────────────────────────────────
    print_header("STEP 5: REGIONAL ESCALATION ALERTS")
    
    escalations = detect_escalations(mandal_alerts, merged, has_mandal)
    
    if escalations.empty:
        print("\n  ✅ No district-level escalations detected.")
    else:
        print(f"\n  🚨 Found {len(escalations)} district-level escalation(s):\n")
        print(escalations.to_string(index=False))
    
    # ── Step 7: Save full report ─────────────────────────────────────────
    print_header("STEP 6: SAVING OUTBREAK REPORT")
    
    all_alerts = []
    for alerts, level in [(pincode_alerts, "Pincode"),
                          (mandal_alerts, "Mandal"),
                          (district_alerts, "District")]:
        if not alerts.empty:
            all_alerts.append(alerts)
    
    if all_alerts:
        report = pd.concat(all_alerts, ignore_index=True)
        report.to_csv(OUTBREAK_REPORT_OUTPUT, index=False)
        print(f"\n  Saved {len(report)} alert(s) to: {OUTBREAK_REPORT_OUTPUT}")
    else:
        print("\n  No alerts to save.")
    
    # ── Summary ──────────────────────────────────────────────────────────
    print_header("SUMMARY")
    print(f"""
  Dataset:
    Records analyzed:  {len(merged):,}
    Unique complaints: {merged['complaint_name'].nunique()}
    Unique pincodes:   {merged['pincode'].nunique()}
    Date range:        {merged['date'].min()} to {merged['date'].max()}
    Districts covered: {merged['district'].nunique()}
    {'Mandals covered:   ' + str(merged['mandal'].nunique()) if has_mandal else ''}
  Alerts:
    Pincode-level:     {len(pincode_alerts)} alert(s)
    {'Mandal-level:      ' + str(len(mandal_alerts)) + ' alert(s)' if has_mandal else ''}
    District-level:    {len(district_alerts)} alert(s)
    Escalations:       {len(escalations)} district outbreak(s)
  Output Files:
    {COMPLAINT_DIR_OUTPUT}
    {OUTBREAK_REPORT_OUTPUT}
""")
if __name__ == "__main__":
    main()
