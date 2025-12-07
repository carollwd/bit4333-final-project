# streamlit_app/app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("Student Performance Prediction App")

# ==============================
# Load models
# ==============================
reg_model_file = "models/best_reg_model.pkl"
clf_model_file = "models/best_clf_model.pkl"

reg_model = joblib.load(reg_model_file)
clf_model = joblib.load(clf_model_file)

# ==============================
# Features setup
# ==============================
# All 27 features exactly as used in training
all_features = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
    "Feature20", "Feature21", "Feature22", "Feature23", "Feature24", "Feature25", "Feature26"
]

# ==============================
# User Inputs
# ==============================
st.subheader("Main Factors (Required)")
hours_studied = st.number_input("Hours Studied per day", min_value=0, max_value=24, value=2)
attendance = st.slider("Attendance (%)", 0, 100, 75)
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
sleep_hours = st.number_input("Sleep Hours per day", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions per week", min_value=0, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

st.subheader("More Factors (Optional)")
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.selectbox("Internet Access", ["No", "Yes"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
physical_activity = st.number_input("Physical Activity per week (hours)", min_value=0, value=3)
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Defaults for remaining features
feature_defaults = [0] * 8  # 27 total - 19 above = 8 remaining

# ==============================
# Prepare input dataframe
# ==============================
input_data = [
    hours_studied, attendance, parental_involvement, access_to_resources,
    extracurricular_activities, sleep_hours, previous_scores, motivation_level,
    internet_access, tutoring_sessions, family_income, teacher_quality,
    school_type, peer_influence, physical_activity, learning_disabilities,
    parental_education_level, distance_from_home, gender
] + feature_defaults

# Convert categorical features to numeric encoding if needed
# Example (simple encoding, match training encoding)
mapping_yes_no = {"No": 0, "Yes": 1}
mapping_low_med_high = {"Low": 0, "Medium": 1, "High": 2}
mapping_neg_neut_pos = {"Negative": 0, "Neutral": 1, "Positive": 2}
mapping_gender = {"Male": 0, "Female": 1}
mapping_school = {"Public": 0, "Private": 1}
mapping_edu = {"High School": 0, "College": 1, "Postgraduate": 2}
mapping_distance = {"Near": 0, "Moderate": 1, "Far": 2}

# Apply encoding
input_data[2] = mapping_low_med_high[input_data[2]]  # Parental Involvement
input_data[3] = mapping_low_med_high[input_data[3]]  # Access to Resources
input_data[4] = mapping_yes_no[input_data[4]]        # Extracurricular
input_data[15] = mapping_yes_no[input_data[15]]      # Learning Disabilities
input_data[8] = mapping_yes_no[input_data[8]]        # Internet Access
input_data[10] = mapping_low_med_high[input_data[10]]  # Family Income
input_data[11] = mapping_low_med_high[input_data[11]]  # Teacher Quality
input_data[12] = mapping_school[input_data[12]]     # School Type
input_data[13] = mapping_neg_neut_pos[input_data[13]]  # Peer Influence
input_data[16] = mapping_edu[input_data[16]]        # Parental Education Level
input_data[17] = mapping_distance[input_data[17]]   # Distance
input_data[18] = mapping_gender[input_data[18]]     # Gender
input_data[7] = mapping_low_med_high[input_data[7]]  # Motivation Level

input_df = pd.DataFrame([input_data], columns=all_features)

# ==============================
# Predictions
# ==============================
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")

        class_pred = clf_model.predict(input_df)[0]
        st.success(f"Predicted Grade/Class: {class_pred}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
