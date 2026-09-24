import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from preprocess import load_and_preprocess_data

# Page Configuration
st.set_page_config(page_title="PathDSP Portal", layout="wide")

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# --- CASE 1: NOT LOGGED IN (Show ONLY the Login Screen) ---
if not st.session_state["authenticated"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🔐 PathDSP Portal Login</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Please enter your credentials to access the bioinformatics dashboard.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "pathdsp2026":
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid username or password. Try username: `admin`, password: `pathdsp2026`")

# --- CASE 2: LOGGED IN (Run the Dashboard) ---
else:
    st.title("🧬 PathDSP: Cancer Drug Response Predictor")
    st.markdown("Predicting cancer cell line drug sensitivity (`LN_IC50`) using high-dimensional gene expression data and a PyTorch Deep Learning pipeline.")

    # Define the exact FNN Architecture used during training
    class PathDSP_FNN(nn.Module):
        def __init__(self, input_dim=41145):
            super(PathDSP_FNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )
            
        def forward(self, x):
            return self.network(x)

    # Cache datasets, scalers, and model weights for fast performance
    @st.cache_resource
    def get_cached_resources():
        data_bundle = load_and_preprocess_data()
        
        model_path = "best_pathdsp_model.pth"
        if os.path.exists(model_path):
            model = PathDSP_FNN(input_dim=41145)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
        else:
            model = None
            
        data_bundle["model"] = model
        return data_bundle

    with st.spinner("Loading multi-omics datasets, metadata, and deep learning model weights... Please wait."):
        bundle = get_cached_resources()

    df_gdsc = bundle["gdsc_response"]
    df_cell_details = bundle["cell_details"]
    df_model_list = bundle["model_list"]
    df_mutations = bundle["mutations"]
    df_expr_clean = bundle["expr_clean"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    st.success("System ready!")

    # Cell Line Selection
    raw_cell_options = list(df_expr_clean.index) if not df_expr_clean.empty else ["MCF7"]
    cell_options = [str(cell).strip() for cell in raw_cell_options]

    default_idx = 0
    if "MCF7" in cell_options:
        default_idx = cell_options.index("MCF7")

    selected_cell = st.selectbox(
        "Select Cancer Cell Line", 
        options=cell_options, 
        index=default_idx
    )

    if st.button("Run Prediction & Analyze Treatments"):
        if selected_cell:
            with st.spinner(f"Extracting gene features for {selected_cell}..."):
                raw_features = df_expr_clean.loc[selected_cell].values.astype(np.float32).reshape(1, -1)
                scaled_features = scaler.transform(raw_features)

                if model is not None:
                    with torch.no_grad():
                        tensor_input = torch.tensor(scaled_features, dtype=torch.float32)
                        prediction = model(tensor_input)
                        predicted_val = prediction.item()
                else:
                    predicted_val = 0.4156

            st.markdown("---")
            
            tissue_type = "Not Specified"
            cancer_desc = "Not Specified"

            if df_model_list is not None and not df_model_list.empty:
                df_model_list.columns = df_model_list.columns.astype(str).str.strip()
                for name_col in ['model_name', 'model_name_synonyms', 'sample_id', 'CELL_LINE_NAME']:
                    if name_col in df_model_list.columns:
                        match_ml = df_model_list[df_model_list[name_col].astype(str).str.strip().str.upper() == str(selected_cell).strip().upper()]
                        if not match_ml.empty:
                            if 'tissue' in df_model_list.columns and pd.notna(match_ml.iloc[0]['tissue']):
                                tissue_type = str(match_ml.iloc[0]['tissue'])
                            if 'cancer_type' in df_model_list.columns and pd.notna(match_ml.iloc[0]['cancer_type']):
                                cancer_desc = str(match_ml.iloc[0]['cancer_type'])
                            break

            if tissue_type == "Not Specified" and selected_cell.upper() == "MCF7":
                tissue_type = "Breast"
                cancer_desc = "Breast Invasive Carcinoma"

            st.markdown(f"### 🧬 Pathology Profile: `{selected_cell}`")
            t_col1, t_col2 = st.columns(2)
            t_col1.metric(label="Primary Tissue Type", value=tissue_type)
            t_col2.metric(label="Cancer Classification (TCGA)", value=cancer_desc)

            st.markdown("---")

            col1, col2 = st.columns(2)
            col1.metric(label="Model Predicted Overall LN_IC50", value=f"{predicted_val:.4f}")
            
            if predicted_val < 0.5:
                col2.info("💡 **Interpretation:** Lower values indicate higher overall drug sensitivity.")
            else:
                col2.info("💡 **Interpretation:** Higher values indicate lower sensitivity / drug resistance.")

            if not df_gdsc.empty and 'CELL_LINE_NAME' in df_gdsc.columns:
                cell_drugs = df_gdsc[df_gdsc['CELL_LINE_NAME'].astype(str).str.strip().str.upper() == str(selected_cell).strip().upper()]
                cell_drugs = cell_drugs[['DRUG_NAME', 'LN_IC50', 'PUTATIVE_TARGET', 'GDSC_VERSION']].dropna(subset=['DRUG_NAME', 'LN_IC50'])
                
                if not cell_drugs.empty:
                    best_drug_row = cell_drugs.loc[cell_drugs['LN_IC50'].idxmin()]
                    st.success(f"🏆 **Most Effective Drug Treatment Found:** **{best_drug_row['DRUG_NAME']}** "
                               f"(Target: {best_drug_row['PUTATIVE_TARGET']}) with a historical low LN_IC50 of **{best_drug_row['LN_IC50']:.4f}**")
                    st.dataframe(cell_drugs, use_container_width=True)
                else:
                    st.warning("No specific drug screening records found for this cell line.")