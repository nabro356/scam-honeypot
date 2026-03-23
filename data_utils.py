import pandas as pd
import numpy as np
import os
import sys
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IP_DATASET_PATH = os.path.join(SCRIPT_DIR, "ip.csv")
PINCODE_DIR_PATH = os.path.join(SCRIPT_DIR, "pincode_directory.csv")
TARGET_STATE = "ANDHRA PRADESH"
def print_header(title):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
def print_subheader(title):
    print(f"\n--- {title} ---")
def load_ip_dataset(path=None):
    """Load and clean the health complaint (IP) dataset."""
    path = path or IP_DATASET_PATH
    print(f"\nLoading IP dataset from: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    raw_count = len(df)
    print(f"  Raw records: {raw_count:,}")
    # Standardize column names
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
    required = ["health_id", "complaint", "complaint_name", "pincode", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        sys.exit(1)
    # Auto-detect timestamp format
    sample_val = df["timestamp"].dropna().iloc[0] if len(df["timestamp"].dropna()) > 0 else ""
    is_unix = False
    try:
        float(sample_val)
        is_unix = True
    except (ValueError, TypeError):
        pass
    if is_unix:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["datetime"].dt.date
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
    # Report missing
    print(f"\n  Missing/invalid values:")
    print(f"    pincode:        {df['pincode'].isna().sum():,}")
    print(f"    timestamp/date: {df['date'].isna().sum():,}")
    print(f"    complaint_name: {df['complaint_name'].isna().sum():,}")
    df = df.dropna(subset=["pincode", "date", "complaint_name"])
    df = df[df["complaint_name"].str.strip() != ""]
    df["pincode"] = df["pincode"].astype(int)
    df["complaint_name"] = df["complaint_name"].str.strip().str.title()
    df["complaint"] = df["complaint"].astype(str).str.strip()
    print(f"\n  Clean records: {len(df):,} (dropped {raw_count - len(df):,})")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique complaints: {df['complaint_name'].nunique()}")
    print(f"  Unique pincodes: {df['pincode'].nunique()}")
    return df
def load_pincode_mapping(path=None):
    """Load India Post pincode directory, return (mapping_df, has_mandal)."""
    path = path or PINCODE_DIR_PATH
    if not os.path.exists(path):
        print(f"WARNING: Pincode directory not found: {path}")
        return None, False
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            pin_df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        return None, False
    col_map = {}
    for col in pin_df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("pincode", "pin_code", "pin"):
            col_map[col] = "pincode"
        elif cl in ("taluk", "taluka", "taluka_name", "taluk_name", "mandal", "tehsil"):
            col_map[col] = "mandal"
        elif cl in ("divisionname", "division_name", "division"):
            col_map[col] = "division"
        elif cl in ("district", "district_name", "districtname"):
            col_map[col] = "district"
        elif cl in ("statename", "state_name", "state"):
            col_map[col] = "state"
    pin_df = pin_df.rename(columns=col_map)
    if "pincode" not in pin_df.columns or "district" not in pin_df.columns:
        return None, False
    has_mandal = False
    if "mandal" in pin_df.columns:
        has_mandal = True
    elif "division" in pin_df.columns:
        pin_df["mandal"] = pin_df["division"]
        has_mandal = True
    # Check for separate mandal mapping
    mandal_csv = os.path.join(SCRIPT_DIR, "pincode_mandal_mapping.csv")
    if os.path.exists(mandal_csv):
        mandal_map = pd.read_csv(mandal_csv)
        mandal_map.columns = [c.strip().lower().replace(" ", "_") for c in mandal_map.columns]
        if "pincode" in mandal_map.columns and "mandal" in mandal_map.columns:
            pin_df = pin_df.drop(columns=["mandal"], errors="ignore")
            pin_df = pin_df.merge(mandal_map[["pincode", "mandal"]], on="pincode", how="left")
            has_mandal = True
    # Filter for state
    if "state" in pin_df.columns:
        pin_df["state"] = pin_df["state"].astype(str).str.strip().str.upper()
        state_mask = pin_df["state"].str.contains("ANDHRA", case=False, na=False)
        if state_mask.sum() > 0:
            pin_df = pin_df[state_mask]
    pin_df["pincode"] = pd.to_numeric(pin_df["pincode"], errors="coerce")
    pin_df = pin_df.dropna(subset=["pincode"])
    pin_df["pincode"] = pin_df["pincode"].astype(int)
    pin_df["district"] = pin_df["district"].astype(str).str.strip().str.title()
    if has_mandal:
        pin_df["mandal"] = pin_df["mandal"].astype(str).str.strip().str.title()
        pin_df.loc[pin_df["mandal"].isin(["Nan", "None", ""]), "mandal"] = "Unknown"
    group_cols = ["pincode", "mandal", "district"] if has_mandal else ["pincode", "district"]
    mapping = pin_df[group_cols].drop_duplicates(subset=["pincode"]).reset_index(drop=True)
    return mapping, has_mandal
def merge_geography(ip_df, pin_mapping, has_mandal):
    """Merge IP data with pincode→mandal/district mapping."""
    if pin_mapping is None:
        ip_df["district"] = "Unknown"
        if has_mandal:
            ip_df["mandal"] = "Unknown"
        return ip_df
    merged = ip_df.merge(pin_mapping, on="pincode", how="left")
    merged["district"] = merged["district"].fillna("Unknown")
    if has_mandal:
        merged["mandal"] = merged["mandal"].fillna("Unknown")
    mapped = (merged["district"] != "Unknown").sum()
    print(f"  Pincode mapping coverage: {mapped:,}/{len(merged):,} ({100*mapped/len(merged):.1f}%)")
    return merged
def build_daily_series(df, region_col, disease_col="complaint_name"):
    """
    Build daily case count time series for each (disease, region) pair.
    Fills missing dates with 0.
    Returns a DataFrame with columns: [region_col, disease_col, date, case_count]
    """
    agg = (
        df.groupby([region_col, disease_col, "date"])
        .agg(case_count=("health_id", "count"))
        .reset_index()
    )
    # Fill missing dates with 0 for each (region, disease) pair
    all_series = []
    for (region, disease), group in agg.groupby([region_col, disease_col]):
        date_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        full = pd.DataFrame({"date": date_range.date})
        full = full.merge(group, on="date", how="left")
        full["case_count"] = full["case_count"].fillna(0).astype(int)
        full[region_col] = region
        full[disease_col] = disease
        all_series.append(full)
    if all_series:
        return pd.concat(all_series, ignore_index=True)
    return pd.DataFrame()
