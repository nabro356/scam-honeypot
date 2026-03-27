import pandas as pd
import numpy as np
def apply_advanced_disease_mapping(ip_df, disease_rules_source, snomed_type_source):
    """
    Applies the 50% intersection matching logic AND filters out 'others' types.
    
    Args:
        ip_df: DataFrame with health data (needs a column with comma-separated SNOMED codes)
        disease_rules_source: Path or DataFrame defining primary diseases.
                              Columns: 'disease_name', 'icd-10', 'symptomn_snomed_codes', 'type'
        snomed_type_source:   Path or DataFrame defining SNOMED to type mappings.
                              Columns: 'snomed_code', 'type'
    
    Returns:
        Mapped DataFrame with records classified as 'others' dropped.
    """
    print("\n--- Applying Advanced Disease CT Code Mapping ---")
    
    # ---------------------------------------------------------
    # 1. LOAD PRIMARY DISEASE RULES
    # ---------------------------------------------------------
    if isinstance(disease_rules_source, str):
        rule_df = pd.read_csv(disease_rules_source)
    else:
        rule_df = disease_rules_source.copy()
        
    rule_df.columns = [c.strip().lower() for c in rule_df.columns]
    
    name_col = [c for c in rule_df.columns if "disease_name" in c or "name" in c][0]
    icd_col = [c for c in rule_df.columns if "icd-10" in c or "icd" in c][0]
    codes_col = [c for c in rule_df.columns if "symptomn" in c or "snomed" in c or "code" in c][0]
    type_col = [c for c in rule_df.columns if "type" in c][0]
    
    disease_rules = {}
    for _, row in rule_df.iterrows():
        disease = str(row[name_col]).strip()
        icd = str(row[icd_col]).strip()
        d_type = str(row[type_col]).strip().lower()
        
        codes_str = str(row[codes_col])
        codes_set = {c.strip() for c in codes_str.replace(";", ",").split(",") if c.strip()}
        
        if len(codes_set) > 0:
            disease_rules[disease] = {
                "icd": icd,
                "type": d_type,
                "codes": codes_set
            }
    
    print(f"  Loaded {len(disease_rules)} primary disease rules.")
    # ---------------------------------------------------------
    # 2. LOAD SECONDARY SNOMED TYPE MAPPING (for <50% fallback)
    # ---------------------------------------------------------
    if isinstance(snomed_type_source, str):
        type_df = pd.read_csv(snomed_type_source)
    else:
        type_df = snomed_type_source.copy()
        
    type_df.columns = [c.strip().lower() for c in type_df.columns]
    
    snomed_col_sec = [c for c in type_df.columns if "snomed" in c or "code" in c][0]
    type_col_sec = [c for c in type_df.columns if "type" in c][0]
    
    # Map SNOMED code -> Type
    snomed_to_type = {}
    for _, row in type_df.iterrows():
        code = str(row[snomed_col_sec]).strip()
        s_type = str(row[type_col_sec]).strip().lower()
        if code:
            snomed_to_type[code] = s_type
            
    print(f"  Loaded {len(snomed_to_type)} individual SNOMED type mappings.")
    # ---------------------------------------------------------
    # 3. PREPARE INPUT DATAFRAME
    # ---------------------------------------------------------
    df = ip_df.copy()
    
    code_field = None
    for col in ["diagnosis", "complaint", "snomed_code", "ct_code"]:
        if col in df.columns:
            code_field = col
            break
            
    if not code_field:
        raise ValueError("Could not find a 'diagnosis' or 'complaint' column in the data.")
        
    print(f"  Processing {len(df):,} patient records (using column '{code_field}')...")
    
    new_rows = []
    stats = {
        "matched_valid": 0, 
        "matched_others_dropped": 0, 
        "fallback_kept": 0, 
        "fallback_others_dropped": 0
    }
    
    # ---------------------------------------------------------
    # 4. EVALUATE EACH PATIENT
    # ---------------------------------------------------------
    for _, row in df.iterrows():
        raw_codes = str(row[code_field])
        patient_codes = {c.strip() for c in raw_codes.replace(";", ",").split(",") if c.strip()}
        
        if not patient_codes:
            # If empty, just keep it or drop it? Usually keep.
            new_rows.append(row)
            stats["fallback_kept"] += 1
            continue
            
        matched_diseases = []
        
        # A. Primary Rule Matching
        for disease_name, rule_data in disease_rules.items():
            required_codes = rule_data["codes"]
            matching_codes = patient_codes & required_codes
            match_count = len(matching_codes)
            
            threshold = len(required_codes) * 0.50
            if match_count >= threshold:
                matched_diseases.append((disease_name, rule_data))
                
        # B. Handle Results
        if len(matched_diseases) == 0:
            # NO MATCH (< 50%). We fall back to the Secondary CSV.
            # Count how many of the patient's codes are mapped to 'others'
            others_count = 0
            for pc in patient_codes:
                if snomed_to_type.get(pc, "") == "others":
                    others_count += 1
            
            # 50% rule for the secondary type checking
            if others_count >= (len(patient_codes) * 0.50):
                # Patient has 50%+ codes that are 'others'. Drop them.
                stats["fallback_others_dropped"] += 1
            else:
                # Keep them (leave old diagnosis and diagnosis_name as they were)
                new_rows.append(row)
                stats["fallback_kept"] += 1
                
        else:
            # MATCHED >= 1 DISEASE.
            # For each matched disease, check if its type is "others".
            for disease_name, rule_data in matched_diseases:
                if rule_data["type"] == "others":
                    # Drop this match
                    stats["matched_others_dropped"] += 1
                else:
                    # VALID match. Update row fields.
                    new_row = row.copy()
                    new_row["complaint_name"] = disease_name
                    new_row["complaint"] = rule_data["icd"]
                    new_row["diagnosis_name"] = disease_name
                    new_row["diagnosis"] = rule_data["icd"]
                    new_rows.append(new_row)
                    stats["matched_valid"] += 1
    final_df = pd.DataFrame(new_rows)
    
    print("\n  Mapping complete!")
    print(f"    Primary matches (Valid kept):     {stats['matched_valid']:,}")
    print(f"    Primary matches ('others' Drop):  {stats['matched_others_dropped']:,}")
    print(f"    Fallback (<50%) Kept:             {stats['fallback_kept']:,}")
    print(f"    Fallback (<50%) ('others' Drop):  {stats['fallback_others_dropped']:,}")
    print(f"  Final mapped dataset size: {len(final_df):,} rows")
    
    return final_df
