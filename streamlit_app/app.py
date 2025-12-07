import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD DATA AND MODELS ---
df = pd.read_csv("models/StudentPerformanceFactors_Cleaned.csv")

reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")
le = joblib.load("models/label_encoder.pkl")
all_training_columns = joblib.load("models/all_training_columns.pkl")

# --- TITLE ---
st.title("Student Performance Predictor")

# --- MAIN FEATURES INPUT ---
main_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 
                 'Motivation_Level', 'Parental_Involvement', 'Sleep_Hours']

user_inputs = {}
st.subheader("Main Features")
for col in main_features:
    if df[col].dtype == 'object':
        user_inputs[col] = st.selectbox(col, df[col].unique())
    else:
        user_inputs[col] = st.number_input(col, value=float(df[col].median()))

# --- OPTIONAL FEATURES ---
optional_features = [c for c in df.columns if c not in main_features + ['Exam_Score','GradeCategory']]
st.subheader("Optional Features (Collapse)")
with st.expander("Show/Hide Optional Features"):
    for col in optional_features:
        if df[col].dtype == 'object':
            user_inputs[col] = st.selectbox(col, df[col].unique())
        else:
            user_inputs[col] = st.number_input(col, value=float(df[col].median()))

# --- PREDICT BUTTON ---
if st.button("Predict"):
    # Create dataframe from user input
    input_df = pd.DataFrame([user_inputs])

    # Encode categorical variables same as training
    input_encoded = pd.get_dummies(input_df)
    for col in all_training_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[all_training_columns]

    # Scale numeric features
    input_scaled = scaler_reg.transform(input_encoded)

    # Regression prediction
    pred_score = reg_model.predict(input_scaled)[0]

    # Convert score to grade
    def score_to_grade(score):
        if score >= 90: return 'A+'
        elif score >= 80: return 'A'
        elif score >= 70: return 'B'
        elif score >= 60: return 'C'
        else: return 'D'

    pred_grade = score_to_grade(pred_score)

    # Show results
    st.write(f"Predicted Exam Score: {pred_score:.2f}")
    st.write(f"Predicted Grade: {pred_grade}")
