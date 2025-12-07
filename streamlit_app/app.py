import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# ===========================
# LOAD MODELS & DATA
# ===========================
@st.cache_resource
def load_all():
    df = pd.read_csv("models/StudentPerformanceFactors_Cleaned.csv")

    reg_model = joblib.load("models/best_reg_model.pkl")
    clf_model = joblib.load("models/best_clf_model.pkl")
    scaler_reg = joblib.load("models/scaler_reg.pkl")
    scaler_clf = joblib.load("models/scaler_clf.pkl")
    all_training_columns = joblib.load("models/all_training_columns.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    return df, reg_model, clf_model, scaler_reg, scaler_clf, all_training_columns, label_encoder


df, reg_model, clf_model, scaler_reg, scaler_clf, all_training_columns, label_encoder = load_all()

st.title("🎓 Student Performance Prediction App")

st.write("Enter a student's details and get predicted exam score + performance class.")


# ======================================================
#  MAIN INPUT FEATURES  (the ones required)
# ======================================================
st.subheader("Main Required Inputs")

main_cols = ["Study_Hours", "Attendance", "Sleep_Hours", "Physical_Activity", "Tutoring_Sessions", "Internet_Access"]
main_inputs = {}

for col in main_cols:
    main_inputs[col] = st.number_input(
        col, 
        min_value=0.0, 
        max_value=24.0 if col in ["Study_Hours", "Sleep_Hours"] else 100.0,
        value=float(df[col].median())
    )


# ======================================================
#  OPTIONAL FEATURES (Categorical + Extra)
# ======================================================
st.subheader("Optional Inputs")

optional_inputs = {}

for col in df.columns:
    if col not in main_cols and col not in ["Exam_Score", "Performance_Index"]:
        if df[col].dtype == object:  
            optional_inputs[col] = st.selectbox(col, df[col].unique())
        else:
            optional_inputs[col] = st.number_input(col, value=float(df[col].median()))


# ======================================================
#  PREDICTION BUTTON
# ======================================================
if st.button("Predict"):

    # Combine all inputs
    user_data = {**main_inputs, **optional_inputs}
    user_df = pd.DataFrame([user_data])

    # --- Encode like training ---
    user_encoded = pd.get_dummies(user_df)

    # --- Realign columns exactly like training ---
    user_encoded = user_encoded.reindex(columns=all_training_columns, fill_value=0)

    # --- Scale for regression ---
    X_reg_scaled = scaler_reg.transform(user_encoded)

    # --- Scale for classification ---
    X_clf_scaled = scaler_clf.transform(user_encoded)

    # --- Predict Regression (Exam Score) ---
    exam_score = reg_model.predict(X_reg_scaled)[0]

    # --- Prevent weird output > 100 or < 0 ---
    exam_score = min(max(exam_score, 0), 100)

    # --- Predict Class ---
    class_pred = clf_model.predict(X_clf_scaled)
    class_pred_label = label_encoder.inverse_transform(class_pred)[0]

    # ===================
    # DISPLAY RESULT
    # ===================
    st.success("Prediction Complete!")

    st.metric("Predicted Exam Score (%)", f"{exam_score:.2f}")
    st.metric("Predicted Performance Class", class_pred_label)
