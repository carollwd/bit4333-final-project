import streamlit as st
import pandas as pd
import joblib
import numpy as np


# 1. UPLOAD CLEANED DATA

st.sidebar.header("Upload Cleaned Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload your cleaned CSV to continue.")
    st.stop()


# 2. UPLOAD MODELS & PREPROCESSORS

st.sidebar.header("Upload Model Files")
uploaded_scaler = st.sidebar.file_uploader("Upload scaler_reg.pkl", type=["pkl"])
uploaded_model = st.sidebar.file_uploader("Upload best_reg_model.pkl", type=["pkl"])
uploaded_columns = st.sidebar.file_uploader("Upload all_training_columns.pkl", type=["pkl"])

if uploaded_scaler and uploaded_model and uploaded_columns:
    scaler_reg = joblib.load(uploaded_scaler)
    best_reg_model = joblib.load(uploaded_model)
    all_training_columns = joblib.load(uploaded_columns)
else:
    st.warning("Please upload all model files to continue.")
    st.stop()


# 3. APP TITLE

st.title("Student Performance Predictor")


# 4. MAIN FEATURES INPUT

main_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 
                 'Motivation_Level', 'Parental_Involvement', 'Sleep_Hours']

st.subheader("Main Features")
user_inputs = {}
for col in main_features:
    if df[col].dtype == 'object':
        user_inputs[col] = st.selectbox(col, df[col].unique())
    else:
        user_inputs[col] = st.number_input(col, value=float(df[col].median()))


# 5. OPTIONAL FEATURES INPUT

optional_features = [c for c in df.columns if c not in main_features + ['Exam_Score']]
st.subheader("Optional Features (Collapse)")
with st.expander("Show/Hide Optional Features"):
    for col in optional_features:
        if df[col].dtype == 'object':
            user_inputs[col] = st.selectbox(col, df[col].unique())
        else:
            user_inputs[col] = st.number_input(col, value=float(df[col].median()))


# 6. PREDICTION BUTTON

if st.button("Predict"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Apply one-hot encoding to match training
    input_encoded = pd.get_dummies(input_df)
    for col in all_training_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[all_training_columns]

    # Scale numeric features
    input_scaled = scaler_reg.transform(input_encoded)

    # Regression prediction
    pred_score = best_reg_model.predict(input_scaled)[0]

    # Display result
    st.success(f"Predicted Exam Score: {pred_score:.2f}")
