# IMPORTS

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="Student Performance Predictor", layout="wide")


# Utility helpers

def load_or_upload_pkls():
    """
    Try to load PKLs from models/ folder. If not found, ask user to upload.
    Returns: (model, scaler, all_columns) or (None, None, None) on failure
    """
    model = scaler = all_columns = None

    # Try to load from models/ path first
    try:
        model = joblib.load("models/best_reg_model.pkl")
        scaler = joblib.load("models/scaler_reg.pkl")
        all_columns = joblib.load("models/all_training_columns.pkl")
        st.info("Loaded model, scaler, and training columns from models/ folder.")
        return model, scaler, all_columns
    except Exception:
        st.warning("Could not load PKLs from models/. Please upload them below (three files).")

    # File uploader (accept multiple files)
    uploaded_files = st.file_uploader(
        "Upload these files: best_reg_model.pkl, scaler_reg.pkl, all_training_columns.pkl",
        accept_multiple_files=True,
        type=["pkl", "joblib"]
    )

    if uploaded_files:
        names = {f.name: f for f in uploaded_files}
        # Attempt to read each required file by filename
        try:
            if "best_reg_model.pkl" in names:
                model = joblib.load(io.BytesIO(names["best_reg_model.pkl"].getvalue()))
            elif "best_reg_model.joblib" in names:
                model = joblib.load(io.BytesIO(names["best_reg_model.joblib"].getvalue()))
            else:
                st.error("Upload file named best_reg_model.pkl (or .joblib).")
                return None, None, None

            if "scaler_reg.pkl" in names:
                scaler = joblib.load(io.BytesIO(names["scaler_reg.pkl"].getvalue()))
            elif "scaler_reg.joblib" in names:
                scaler = joblib.load(io.BytesIO(names["scaler_reg.joblib"].getvalue()))
            else:
                st.error("Upload file named scaler_reg.pkl (or .joblib).")
                return None, None, None

            if "all_training_columns.pkl" in names:
                all_columns = joblib.load(io.BytesIO(names["all_training_columns.pkl"].getvalue()))
            elif "all_training_columns.joblib" in names:
                all_columns = joblib.load(io.BytesIO(names["all_training_columns.joblib"].getvalue()))
            else:
                st.error("Upload file named all_training_columns.pkl (or .joblib).")
                return None, None, None

            st.success("Uploaded and loaded model, scaler, and training columns.")
            return model, scaler, all_columns

        except Exception as e:
            st.error(f"Failed to load uploaded files: {e}")
            return None, None, None

    return None, None, None


def prepare_input_df(input_df, all_cols):
    """One-hot encode input_df and reindex to all_cols (training columns)."""
    # get_dummies for any categorical columns in input_df
    df_encoded = pd.get_dummies(input_df)
    # add missing columns, remove extras, and order exactly
    df_encoded = df_encoded.reindex(columns=all_cols, fill_value=0)
    return df_encoded



# Load files (model, scaler, columns)

model, scaler, all_training_columns = load_or_upload_pkls()

# If files not available yet, stop UI below uploader
if model is None or scaler is None or all_training_columns is None:
    st.info("Upload required files to proceed.")
    st.stop()


# Load cleaned CSV for reference values (optional)
# try models/StudentPerformanceFactors_Cleaned.csv; if missing show uploader

def load_cleaned_csv():
    try:
        df_ref = pd.read_csv("models/StudentPerformanceFactors_Cleaned.csv")
        return df_ref
    except Exception:
        uploaded = st.file_uploader("Upload cleaned StudentPerformanceFactors_Cleaned.csv (optional but helps dropdowns)", type=["csv"])
        if uploaded:
            df_ref = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            return df_ref
    return None

df_ref = load_cleaned_csv()


# Title + instructions

st.title("Student Performance Predictor (Regression)")
st.write("Predict exam score from student factors. Main features first, others optional in the collapsed panel.")


# Main features (top)
# Keep names consistent with dataset

main_features = [
    "Hours_Studied",
    "Access_to_Resources",
    "Attendance",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Learning_Disabilities"
]

st.subheader("Main features (required)")
user_input = {}

col1, col2, col3 = st.columns(3)

with col1:
    # Hours studied
    user_input["Hours_Studied"] = st.number_input("Hours Studied (per week)", min_value=0.0, max_value=168.0,
                                                value=5.0, step=1.0)
    # Access to resources
    if df_ref is not None and "Access_to_Resources" in df_ref.columns:
        opts = df_ref["Access_to_Resources"].unique().tolist()
        user_input["Access_to_Resources"] = st.selectbox("Access to Resources", opts)
    else:
        user_input["Access_to_Resources"] = st.selectbox("Access to Resources", ["Low", "Medium", "High"])

with col2:
    user_input["Attendance"] = st.slider("Attendance (%)", 0, 100, 75)
    user_input["Sleep_Hours"] = st.number_input("Sleep Hours (per day)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

with col3:
    user_input["Tutoring_Sessions"] = st.number_input("Tutoring Sessions (per month)", min_value=0, value=0, step=1)
    if df_ref is not None and "Learning_Disabilities" in df_ref.columns:
        opts = df_ref["Learning_Disabilities"].unique().tolist()
        user_input["Learning_Disabilities"] = st.selectbox("Learning Disabilities", opts)
    else:
        user_input["Learning_Disabilities"] = st.selectbox("Learning Disabilities", ["No", "Yes"])


# Optional features in a collapsed expander

st.subheader("Optional features")
with st.expander("Show/Hide Optional Features"):
    optional_inputs = {}
    # Determine list of optional columns from training columns minus main features and target if present
    optional_cols = [c for c in all_training_columns if c not in main_features and c != "Exam_Score"]
    # Extract base column names before one-hot suffix for better widget labels
    # We'll attempt to pull unique values from reference df where possible
    for col in optional_cols:
        # If this column is like "Gender_Male" that's a dummy column; we will present the base name as categorical
        if "_" in col:
            base = col.split("_")[0]
        else:
            base = col
        # Avoid repeating widgets for multiple dummies of same base: present once per base
    # Build set of unique bases but preserve order as in optional_cols
    seen = set()
    bases = []
    for col in optional_cols:
        base = col.split("_")[0] if "_" in col else col
        if base not in seen:
            seen.add(base)
            bases.append(base)

    # Create widgets (for bases). We will map these back to values when building final input dict.
    for base in bases:
        # Skip if base is actually a main feature (shouldn't happen)
        if base in main_features or base == "Exam_Score":
            continue

        # If we have reference df and base column exists in it, use its unique values
        if df_ref is not None and base in df_ref.columns:
            vals = df_ref[base].unique().tolist()
            # if numeric, use number_input
            if df_ref[base].dtype.kind in 'iufc':
                optional_inputs[base] = st.number_input(base, value=float(df_ref[base].median()))
            else:
                optional_inputs[base] = st.selectbox(base, vals)
        else:
            # Default fallback: categorical yes/no or simple numeric
            optional_inputs[base] = st.text_input(base, value="")

# Merge optional inputs into user_input (but careful with dummy mapping later)
user_input.update(optional_inputs)

# --------------------
# Predict button
# --------------------
st.write("")  # spacing
if st.button("Predict"):
    try:
        # Convert user_input into DataFrame
        input_df = pd.DataFrame([user_input])

        # Some text inputs might be empty strings; replace with a sensible default if needed
        input_df = input_df.replace({"": np.nan})

        # For any columns that should be numeric but are strings, attempt conversion
        for col in input_df.columns:
            # if column exists in training as numeric (no '_' and present in all_training_columns), try convert
            if col in all_training_columns and col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except Exception:
                    pass  # keep as-is (will be handled by get_dummies)

        # Encode & align with training columns
        input_encoded = prepare_input_df(input_df, all_training_columns)

        # Scale
        try:
            input_scaled = scaler.transform(input_encoded)
        except Exception as e:
            st.error(f"Scaling error: {e}")
            st.stop()

        # Predict
        pred_score = model.predict(input_scaled)[0]
        pred_score = float(pred_score)  # ensure native float

        # Optional grade for display
        pred_grade = score_to_grade(pred_score)

        # Output
        st.success(f"Predicted Exam Score: {pred_score:.2f}")
        st.info(f"Predicted Grade (display only): {pred_grade}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer: small help

st.markdown("---")
st.caption("If your dropdowns look empty, upload the cleaned CSV when prompted so the app can show correct options. The model expects the same preprocessing used during training.")
