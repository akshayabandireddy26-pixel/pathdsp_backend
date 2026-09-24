import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_and_preprocess_data():
    cache_file = "cache_bundle.pkl"
    
    # If a pre-compiled cache exists, load it instantly (< 1 second)
    if os.path.exists(cache_file):
        print("Loading from fast local cache...")
        return joblib.load(cache_file)
    
    print("Building datasets from scratch (first time only)...")
    data_dir = "Data"
    
    # 1. Load GDSC1 & GDSC2 Dose Response Data
    df_gdsc1, df_gdsc2 = None, None
    gdsc1_path = os.path.join(data_dir, "GDSC1_fitted_dose_response_27oct23.xlsx")
    gdsc2_path = os.path.join(data_dir, "GDSC2_fitted_dose_response_27oct23.xlsx")
    
    if os.path.exists(gdsc1_path):
        df_gdsc1 = pd.read_excel(gdsc1_path)
        df_gdsc1['GDSC_VERSION'] = 'GDSC1'
    if os.path.exists(gdsc2_path):
        df_gdsc2 = pd.read_excel(gdsc2_path)
        df_gdsc2['GDSC_VERSION'] = 'GDSC2'
        
    df_gdsc = pd.concat([df_gdsc1, df_gdsc2], ignore_index=True) if df_gdsc1 is not None and df_gdsc2 is not None else (df_gdsc1 if df_gdsc1 is not None else df_gdsc2)

    # 2. Load Metadata files
    df_cell_details = pd.read_excel(os.path.join(data_dir, "Cell_Lines_Details.xlsx")) if os.path.exists(os.path.join(data_dir, "Cell_Lines_Details.xlsx")) else None
    
    df_model_list, df_mutations = None, None
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if "model" in filename.lower() and filename.endswith((".csv", ".xlsx")):
            df_model_list = pd.read_csv(path, low_memory=False) if filename.endswith(".csv") else pd.read_excel(path)
        if "mutation" in filename.lower() and filename.endswith((".csv", ".xlsx")):
            df_mutations = pd.read_csv(path, low_memory=False) if filename.endswith(".csv") else pd.read_excel(path)

    # 3. Load Gene Expression Matrix (RNA-Seq TPM)
    expr_path = os.path.join(data_dir, "rnaseq_merged_rsem_tpm_20260323.csv")
    df_expr = pd.read_csv(expr_path, low_memory=False)
    
    model_names = df_expr.iloc[0, 3:].values
    gene_symbols = df_expr.iloc[3:, 2].values
    expr_matrix = df_expr.iloc[3:, 3:].values.astype(float)

    df_expr_clean = pd.DataFrame(expr_matrix.T, index=model_names, columns=gene_symbols)
    df_expr_clean = df_expr_clean.loc[:, ~df_expr_clean.columns.duplicated()].fillna(0)

    scaler = StandardScaler()
    scaler.fit(df_expr_clean.values.astype(np.float32))

    bundle = {
        "gdsc_response": df_gdsc,
        "cell_details": df_cell_details,
        "model_list": df_model_list,
        "mutations": df_mutations,
        "expr_clean": df_expr_clean,
        "scaler": scaler
    }
    
    # Save cache for lightning-fast future loads
    joblib.dump(bundle, cache_file)
    return bundle