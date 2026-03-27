"""
Disease to CT Code Mapper
=========================
Classifies patient records into standard diseases based on multiple SNOMED/CT codes.
Logic:
  1. A patient can have multiple CT codes (e.g., "111, 222, 333").
  2. A disease definition requires specific CT codes (e.g., Dengue = "111, 222, 444, 555").
  3. We find exactly which codes match (intersection).
  4. If the number of matching codes >= 50% of the disease's required codes,
     the patient is classified as that disease.
  5. If they match NO diseases, their `complaint_name` becomes their FIRST CT code.
  6. If they match MULTIPLE diseases, the row is duplicated for each matched disease.
Usage from notebook:
  from disease_mapper import apply_disease_mapping
  
  # Load your health data from Impala
  ip_df = pd.read_sql("SELECT * FROM your_table", conn)
  
  # Apply mapping (needs a mapping DataFrame or CSV path)
  # mapping_df must have columns: ['disease_name', 'ct_codes'] (comma-separated codes)
  clean_df = apply_disease_mapping(ip_df, mapping_df)
  
  # Pass to Isolation Forest
  from outbreak_iforest_epi import main
  main(ip_df=clean_df)
"""
import pandas as pd
import numpy as np
def apply_disease_mapping(ip_df, mapping_source):
    """
    Applies the 50% intersection matching logic to classify rows.
    
    Args:
        ip_df: The health data DataFrame (needs a column with comma-separated SNOMED codes)
        mapping_source: Path to CSV or a DataFrame defining disease rules.
                        Expected columns: 'disease_name', 'icd-10', 'symptomn_snomed_codes'
    
    Returns:
        A new DataFrame with 'complaint_name' (disease_name) and 'complaint' (icd-10) resolved.
    """
    print("\n--- Applying Disease CT Code Mapping ---")
    
    # 1. Load the rule mapping
    if isinstance(mapping_source, str):
        rule_df = pd.read_csv(mapping_source)
    else:
        rule_df = mapping_source.copy()
        
    # Standardize rule columns
    rule_df.columns = [c.strip().lower() for c in rule_df.columns]
    
    name_col = [c for c in rule_df.columns if "disease_name" in c or "name" in c][0]
    icd_col = [c for c in rule_df.columns if "icd-10" in c or "icd" in c][0]
    codes_col = [c for c in rule_df.columns if "symptomn" in c or "snomed" in c or "code" in c][0]
    
    # Build dictionary of rules
    # Structure: {"Disease Name": {"icd": "A00", "codes": {"111", "222"}}}
    disease_rules = {}
    for _, row in rule_df.iterrows():
        disease = str(row[name_col]).strip()
        icd = str(row[icd_col]).strip()
        
        # Parse comma-separated codes into a Python set
        codes_str = str(row[codes_col])
        codes_set = {c.strip() for c in codes_str.replace(";", ",").split(",") if c.strip()}
        
        if len(codes_set) > 0:
            disease_rules[disease] = {
                "icd": icd,
                "codes": codes_set
            }
            
    print(f"  Loaded {len(disease_rules)} disease definitions.")
        
    df = ip_df.copy()
    
    # Identify which column has the codes (usually 'complaint', 'diagnosis', 'snomed_code')
    code_field = None
    for col in ["diagnosis", "complaint", "snomed_code", "ct_code"]:
        if col in df.columns:
            code_field = col
            break
            
    if not code_field:
        raise ValueError("Could not find a 'diagnosis' or 'complaint' column in the data.")
        
    print(f"  Processing {len(df):,} patient records (using column '{code_field}')...")
    
    new_rows = []
    stats = {"matched_one": 0, "matched_multiple": 0, "matched_none_fallback": 0}
    
    for _, row in df.iterrows():
        raw_codes = str(row[code_field])
        
        # Parse patient's codes into a set
        patient_codes = {c.strip() for c in raw_codes.replace(";", ",").split(",") if c.strip()}
        
        if not patient_codes:
            new_rows.append(row)
            stats["matched_none_fallback"] += 1
            continue
            
        matched_diseases = []
        
        # Evaluate against every disease rule
        for disease_name, rule_data in disease_rules.items():
            required_codes = rule_data["codes"]
            
            # EXACT MATCHING: intersection
            matching_codes = patient_codes & required_codes
            match_count = len(matching_codes)
            
            # Threshold is 50%
            threshold = len(required_codes) * 0.50
            
            if match_count >= threshold:
                matched_diseases.append((disease_name, rule_data["icd"]))
                
        # Handle results
        if len(matched_diseases) == 0:
            # FALLBACK: Matched none. Leave it exactly as it is.
            new_rows.append(row)
            stats["matched_none_fallback"] += 1
            
        elif len(matched_diseases) == 1:
            # Matched exactly one disease
            disease_name, icd = matched_diseases[0]
            new_row = row.copy()
            # Standardize output for data_utils / isolation forest
            new_row["complaint_name"] = disease_name
            new_row["complaint"] = icd
            # Also overwrite original raw fields so they don't leak
            new_row["diagnosis_name"] = disease_name
            new_row["diagnosis"] = icd
            new_rows.append(new_row)
            stats["matched_one"] += 1
            
        else:
            # Matched MULTIPLE diseases. Duplicate row.
            for disease_name, icd in matched_diseases:
                new_row = row.copy()
                new_row["complaint_name"] = disease_name
                new_row["complaint"] = icd
                new_row["diagnosis_name"] = disease_name
                new_row["diagnosis"] = icd
                new_rows.append(new_row)
            stats["matched_multiple"] += 1
            
    final_df = pd.DataFrame(new_rows)
    
    print("\n  Mapping complete!")
    print(f"    Records that matched 1 disease:         {stats['matched_one']:,}")
    print(f"    Records that matched multiple diseases: {stats['matched_multiple']:,}")
    print(f"    Records that matched none (fallback):   {stats['matched_none_fallback']:,}")
    print(f"  Final mapped dataset size: {len(final_df):,} rows")
    
    return final_df
