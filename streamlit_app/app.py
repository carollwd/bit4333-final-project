import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load models
# =========================
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
le = joblib.load("models/label_encoder.pkl")  # For classification target

# =========================
# Upload CSV
# =========================
st.sidebar.header("Upload Cleaned CSV")
uploaded_file = st.sidebar.file_uploader("Choose cleaned CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload the cleaned CSV to continue.")
    st.stop()

# =========================
# Features
# =========================
target_col = "Exam_Score"
all_features = [c for c in df.columns if c != target_col]
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# 6 main features
main_features = ["Hours_Studied", "Access_to_Resources", "Attendance",
                 "Sleep_Hours", "Tutoring_Sessions", "Learning_Disabilities"]

st.title("Student Performance Prediction")
st.write("""
Predict student's exam score and grade.
Main 6 features are required; optional features for more accuracy.
""")

# --------------------------
# Main 6 inputs
# --------------------------
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
access_to_resources = st.selectbox("Access to Resources", df["Access_to_Resources"].unique())
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", df["Learning_Disabilities"].unique())

input_data = {
    "Hours_Studied": hours_studied,
    "Access_to_Resources": access_to_resources,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Learning_Disabilities": learning_disabilities
}

# --------------------------
# Optional features
# --------------------------
with st.expander("Optional Features"):
    optional_inputs = {}
    for col in all_features:
        if col in main_features:
            continue
        if col in categorical_features:
            optional_inputs[col] = st.selectbox(col, df[col].unique())
        else:
            optional_inputs[col] = st.number_input(col, value=int(df[col].median()))
    input_data.update(optional_inputs)

# --------------------------
# Convert input to DataFrame
# --------------------------
input_df = pd.DataFrame([input_data])

# Encode categorical features
for col in categorical_features:
    if col in input_df.columns:
        input_df[col] = pd.Categorical(input_df[col],
                                       categories=df[col].unique()).codes

# Ensure all features are present and in correct order
missing_cols = set(all_features) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0  # default if missing

input_df = input_df[all_features]

# --------------------------
# Prediction
# --------------------------
if st.button("Predict"):
    try:
        # Regression
        X_reg_scaled = scaler_reg.transform(input_df)
        exam_score = reg_model.predict(X_reg_scaled)[0]
        exam_score = max(0, min(100, exam_score))  # clamp 0-100
        st.success(f"Predicted Exam Score: {exam_score:.2f}")

        # Classification
        X_clf_scaled = scaler_clf.transform(input_df)
        class_pred = clf_model.predict(X_clf_scaled)[0]
        class_label = le.inverse_transform([class_pred])[0]
        st.success(f"Predicted Grade: {class_label}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
