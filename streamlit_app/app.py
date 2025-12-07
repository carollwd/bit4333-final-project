import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load models
# ==============================
reg_model_file = "models/best_reg_model.pkl"
clf_model_file = "models/best_clf_model.pkl"

reg_model = joblib.load(reg_model_file)
clf_model = joblib.load(clf_model_file)

# ==============================
# Load training data CSV for feature reference
# ==============================
train_df = pd.read_csv("data/StudentPerformanceFactors_Cleaned.csv")

target_col = "Exam_Score"
all_features = [c for c in train_df.columns if c != target_col]

categorical_features = [
    'Access_to_Resources', 'Parental_Involvement', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

numeric_features = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions',
    'Previous_Scores', 'Physical_Activity'
]

# Main 6 features
main_features = [
    "Hours_Studied", "Access_to_Resources", "Attendance",
    "Sleep_Hours", "Tutoring_Sessions", "Learning_Disabilities"
]

optional_features = [f for f in all_features if f not in main_features]

# ==============================
# Streamlit UI
# ==============================
st.title("Student Performance Prediction")

st.write("""
Predict student's exam score and performance classification.
Main 6 features are required; optional features are under "More Factors".
""")

# ------------------------------
# Main 6 inputs
# ------------------------------
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
access_to_resources = st.selectbox("Access to Resources", train_df["Access_to_Resources"].unique())
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", train_df["Learning_Disabilities"].unique())

# ------------------------------
# Optional features in expander
# ------------------------------
optional_inputs = {}
with st.expander("More Factors (Optional)"):
    for col in optional_features:
        if col in categorical_features:
            optional_inputs[col] = st.selectbox(col, train_df[col].unique())
        else:
            optional_inputs[col] = st.number_input(col, value=int(train_df[col].median()))

# ------------------------------
# Prepare input DataFrame
# ------------------------------
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

# ==============================
# Encode categorical features exactly like training
# ==============================
for col in categorical_features:
    if col in input_df.columns:
        input_df[col] = pd.Categorical(
            input_df[col],
            categories=train_df[col].unique()
        ).codes
    else:
        input_df[col] = -1  # fill missing categorical features

# ==============================
# Ensure all features exist and are in correct order
# ==============================
for col in all_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[all_features]

# ==============================
# Prediction
# ==============================
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")
    except Exception as e:
        st.error(f"Regression prediction failed: {e}")

    try:
        class_pred = clf_model.predict(input_df)[0]
        st.success(f"Predicted Performance Class: {class_pred}")
    except Exception as e:
        st.error(f"Classification prediction failed: {e}")

