import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load models
# -------------------------------
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

# -------------------------------
# Define all features (27 total)
# -------------------------------
all_features = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
    "Feature_20", "Feature_21", "Feature_22", "Feature_23", "Feature_24",
    "Feature_25", "Feature_26"
]

# -------------------------------
# Helper for mapping categorical to numeric
# -------------------------------
def map_categorical(val, mapping):
    return mapping.get(val, 0)  # default 0 if not found

# Example mappings for categorical variables
yes_no_map = {"Yes": 1, "No": 0}
level_map = {"Low": 0, "Medium": 1, "High": 2}
gender_map = {"Male": 0, "Female": 1}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Student Performance Predictor")

st.header("Main Factors (required)")
hours_studied = st.number_input("Hours Studied", 0, 24, 6)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
attendance = st.slider("Attendance (%)", 0, 100, 80)
sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
tutoring_sessions = st.number_input("Tutoring Sessions", 0, 10, 2)
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])

st.header("More Factors (optional)")
# You can repeat similar inputs for optional features if you want
# If left blank, they will default to 0

# -------------------------------
# Prepare input dataframe
# -------------------------------
input_dict = {f: [0] for f in all_features}

# Fill main 6
input_dict["Hours_Studied"] = [hours_studied]
input_dict["Access_to_Resources"] = [map_categorical(access_to_resources, level_map)]
input_dict["Attendance"] = [attendance]
input_dict["Sleep_Hours"] = [sleep_hours]
input_dict["Tutoring_Sessions"] = [tutoring_sessions]
input_dict["Learning_Disabilities"] = [map_categorical(learning_disabilities, yes_no_map)]

# Optional: fill more factors if user provides
# Example: input_dict["Parental_Involvement"] = [map_categorical(st.selectbox(...), level_map)]

input_df = pd.DataFrame(input_dict)

# -------------------------------
# Make predictions
# -------------------------------
try:
    exam_score = reg_model.predict(input_df)[0]
    st.success(f"Predicted Exam Score: {exam_score:.2f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")

try:
    risk_class = clf_model.predict(input_df)[0]
    st.success(f"Predicted Risk/Class: {risk_class}")
except Exception as e:
    st.error(f"Classification failed: {e}")
