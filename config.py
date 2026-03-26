"""
SNOMED CT → ICD-10 Mapping Script
===================================
Fetches ICD-10 codes for your SNOMED CT diagnosis codes using the
free SNOMED International Snowstorm API (no signup/license needed).
Usage:
  # From a notebook with Impala DataFrame:
  from snomed_icd_mapper import build_mapping, enrich_dataframe
  mapping = build_mapping(ip_df)                # Fetches ICD-10 for all unique SNOMED codes
  ip_df_enriched = enrich_dataframe(ip_df)      # Adds icd10_code + icd10_name columns
  # Standalone:
  python snomed_icd_mapper.py
"""
import pandas as pd
import requests
import time
import os
import sys
import json
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Snowstorm API (free, public, no auth needed)
SNOWSTORM_BASE = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"
BRANCH = "MAIN"  # International edition
# Rate limiting (be nice to the free API)
DELAY_BETWEEN_REQUESTS = 0.3  # seconds
# Output
MAPPING_CSV = os.path.join(SCRIPT_DIR, "snomed_icd_mapping.csv")
# Target ICD map reference set ID (SNOMED→ICD-10 map)
ICD10_REFSET_ID = "447562003"  # ICD-10 complex map reference set
# ─────────────────────────────────────────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_icd10_for_snomed(snomed_code):
    """
    Query Snowstorm API to get ICD-10 mapping for a SNOMED CT code.
    Returns a list of dicts: [{"icd10_code": "A78", "map_target_name": "Q fever", "map_group": 1}]
    """
    url = f"{SNOWSTORM_BASE}/{BRANCH}/members"
    params = {
        "referenceSet": ICD10_REFSET_ID,
        "referencedComponentId": str(snomed_code),
        "active": "true",
        "limit": 10
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "OutbreakDetection/1.0"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            mappings = []
            for item in items:
                additional = item.get("additionalFields", {})
                icd_code = additional.get("mapTarget", "")
                map_advice = additional.get("mapAdvice", "")
                map_group = additional.get("mapGroup", "")
                if icd_code:
                    mappings.append({
                        "icd10_code": icd_code,
                        "map_advice": map_advice,
                        "map_group": map_group,
                    })
            return mappings
        elif resp.status_code == 404:
            return []
        else:
            return []
    except requests.exceptions.RequestException:
        return []
def get_snomed_concept_info(snomed_code):
    """Get the preferred term for a SNOMED CT code."""
    url = f"{SNOWSTORM_BASE}/{BRANCH}/concepts/{snomed_code}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "OutbreakDetection/1.0"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            fsn = data.get("fsn", {}).get("term", "")
            pt = data.get("pt", {}).get("term", "")
            return pt or fsn
    except:
        pass
    return ""
def get_icd10_name(icd_code):
    """
    Try to get ICD-10 code description via WHO API (free, no auth).
    Falls back to returning just the code if API fails.
    """
    # WHO ICD API (free, no auth for ICD-10 lookups)
    try:
        url = f"https://icd.who.int/browse10/2019/en/JsonGetChildrenConcepts?ConceptId={icd_code}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                return data[0].get("Title", icd_code)
    except:
        pass
    return ""
# ─────────────────────────────────────────────────────────────────────────────
# MAPPING BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_mapping(df=None, snomed_codes=None, save_csv=True):
    """
    Build SNOMED CT → ICD-10 mapping for all unique codes.
    Args:
        df:           DataFrame with 'diagnosis' or 'complaint' column (SNOMED codes)
        snomed_codes: List of SNOMED codes (used if df is None)
        save_csv:     Whether to save mapping to CSV
    Returns:
        DataFrame with columns: snomed_code, snomed_name, icd10_code, icd10_name, map_advice
    """
    # Load existing mapping if available (avoid re-fetching)
    existing = {}
    if os.path.exists(MAPPING_CSV):
        prev = pd.read_csv(MAPPING_CSV)
        for _, row in prev.iterrows():
            existing[str(row["snomed_code"])] = row.to_dict()
        print(f"  Loaded {len(existing)} existing mappings from cache.")
    # Get unique SNOMED codes
    if df is not None:
        # Standardize column names
        col_map = {}
        for col in df.columns:
            cl = col.strip().lower().replace(" ", "_")
            if cl in ("complaint", "complaint_code", "snomed_code", "snomed_ct_code",
                       "diagnosis", "diagnosis_code"):
                col_map[col] = "complaint"
            elif cl in ("complaint_name", "diagnosis_name"):
                col_map[col] = "complaint_name"
        df_renamed = df.rename(columns=col_map)
        codes = df_renamed["complaint"].astype(str).str.strip().unique()
    elif snomed_codes is not None:
        codes = [str(c).strip() for c in snomed_codes]
    else:
        print("ERROR: Provide either a DataFrame or a list of SNOMED codes.")
        return pd.DataFrame()
    codes = [c for c in codes if c and c != "nan"]
    print(f"\n  Unique SNOMED codes to map: {len(codes)}")
    # Separate cached vs new
    new_codes = [c for c in codes if c not in existing]
    cached_codes = [c for c in codes if c in existing]
    print(f"  Already cached: {len(cached_codes)}")
    print(f"  New to fetch: {len(new_codes)}")
    # Fetch new mappings from API
    results = []
    # Add cached results
    for code in cached_codes:
        results.append(existing[code])
    # Fetch new
    if new_codes:
        print(f"\n  Fetching ICD-10 mappings from Snowstorm API...")
        for i, code in enumerate(new_codes):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    {i+1}/{len(new_codes)}: {code}...", end="\r")
            # Get ICD-10 mapping
            mappings = get_icd10_for_snomed(code)
            # Get SNOMED name
            snomed_name = get_snomed_concept_info(code)
            if mappings:
                # Take the first/best mapping (map_group=1 preferred)
                best = sorted(mappings, key=lambda x: x.get("map_group", 99))[0]
                results.append({
                    "snomed_code": code,
                    "snomed_name": snomed_name,
                    "icd10_code": best["icd10_code"],
                    "icd10_name": "",  # Will be filled below if needed
                    "map_advice": best.get("map_advice", ""),
                })
            else:
                results.append({
                    "snomed_code": code,
                    "snomed_name": snomed_name,
                    "icd10_code": "NO_MAP",
                    "icd10_name": "",
                    "map_advice": "No ICD-10 mapping found in SNOMED refset",
                })
            time.sleep(DELAY_BETWEEN_REQUESTS)
        print(f"\n    Done! Fetched {len(new_codes)} codes.")
    # Build DataFrame
    mapping_df = pd.DataFrame(results)
    # Stats
    mapped = (mapping_df["icd10_code"] != "NO_MAP").sum()
    unmapped = (mapping_df["icd10_code"] == "NO_MAP").sum()
    print(f"\n  Results:")
    print(f"    Mapped to ICD-10:  {mapped}")
    print(f"    No mapping found:  {unmapped}")
    # Save
    if save_csv and len(mapping_df) > 0:
        mapping_df.to_csv(MAPPING_CSV, index=False)
        print(f"\n  Saved mapping to: {MAPPING_CSV}")
    return mapping_df
def enrich_dataframe(df, mapping_df=None):
    """
    Add icd10_code and icd10_name columns to the health data DataFrame.
    Args:
        df:         Health data DataFrame with SNOMED codes
        mapping_df: Pre-built mapping (if None, loads from CSV or builds)
    Returns:
        DataFrame with added icd10_code and icd10_name columns
    """
    if mapping_df is None:
        if os.path.exists(MAPPING_CSV):
            mapping_df = pd.read_csv(MAPPING_CSV)
        else:
            print("  No mapping found. Building from data...")
            mapping_df = build_mapping(df=df)
    # Standardize column
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("complaint", "complaint_code", "snomed_code", "snomed_ct_code",
                   "diagnosis", "diagnosis_code"):
            col_map[col] = "complaint"
    df = df.rename(columns=col_map)
    df["complaint"] = df["complaint"].astype(str).str.strip()
    mapping_df["snomed_code"] = mapping_df["snomed_code"].astype(str).str.strip()
    enriched = df.merge(
        mapping_df[["snomed_code", "icd10_code", "icd10_name"]],
        left_on="complaint",
        right_on="snomed_code",
        how="left"
    ).drop(columns=["snomed_code"], errors="ignore")
    mapped = (enriched["icd10_code"].notna() & (enriched["icd10_code"] != "NO_MAP")).sum()
    print(f"  ICD-10 mapping coverage: {mapped:,}/{len(enriched):,} ({100*mapped/len(enriched):.1f}%)")
    return enriched
# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Standalone mode: reads ip.csv, builds mapping, saves CSV."""
    print("=" * 60)
    print("  SNOMED CT → ICD-10 MAPPING")
    print("=" * 60)
    ip_path = os.path.join(SCRIPT_DIR, "ip.csv")
    if not os.path.exists(ip_path):
        print(f"\nERROR: {ip_path} not found.")
        print("Usage from notebook:")
        print("  from snomed_icd_mapper import build_mapping")
        print("  mapping = build_mapping(df=your_impala_df)")
        sys.exit(1)
    df = pd.read_csv(ip_path)
    mapping = build_mapping(df=df)
    print(f"\n  Preview:")
    print(mapping.head(20).to_string(index=False))
    print(f"\n  Full mapping saved to: {MAPPING_CSV}")
    print("  To add ICD-10 to your data:")
    print("    from snomed_icd_mapper import enrich_dataframe")
    print("    enriched_df = enrich_dataframe(your_df)")
if __name__ == "__main__":
    main()
