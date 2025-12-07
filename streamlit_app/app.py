import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and training features
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
all_features = joblib.load("models/all_features.pkl")

# Load cleaned dataset for dropdowns / defaults
df = pd.read_csv("data/StudentPerformanceFactors_Cleaned.csv")
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

st.title("Student Performance Prediction")
st.write("Predict exam score and performance class. Fill main features first, optional below.")

# Main 6 inputs
hours_studied = st.number_input("Hours Studied", 0, 24, 5)
access_to_resources = st.selectbox("Access to Resources", ["Low","Medium","High"])
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
tutoring_sessions = st.number_input("Tutoring Sessions", 0, 0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No","Yes"])

# Optional features
with st.expander("Optional Features"):
    optional_inputs = {}
    for col in all_features:
        if col in ["Hours_Studied","Access_to_Resources","Attendance",
                   "Sleep_Hours","Tutoring_Sessions","Learning_Disabilities"]:
            continue
        if col in categorical_features:
            optional_inputs[col] = st.selectbox(col, df[col].unique())
        else:
            optional_inputs[col] = st.number_input(col, value=int(df[col].median()))

# Prepare input dataframe
input_data = {
    "Hours_Studied": hours_studied,
    "Access_to_Resources": access_to_resources,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Learning_Disabilities": learning_disabilities
}
input_data.update(optional_inputs)
input_df = pd.DataFrame([input_data])

# Encode categorical features
for col in categorical_features:
    if col in input_df.columns:
        input_df[col] = pd.Categorical(input_df[col], categories=df[col].unique()).codes

# Add missing features
for col in all_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[all_features]

# Prediction
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")
    except:
        st.error("Regression failed")

    try:
        class_pred = clf_model.predict(input_df)[0]
        st.success(f"Predicted Performance Class: {class_pred}")
    except:
        st.error("Classification failed")
