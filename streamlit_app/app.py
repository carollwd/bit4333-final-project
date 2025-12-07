import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

st.title("Student Performance Prediction")

st.markdown("## Main Factors (Required)")
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

st.markdown("## More Factors (Optional)")
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=60)
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
internet_access = st.selectbox("Internet Access", ["No", "Yes"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=1)
physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=50, value=3)
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Collect input into a DataFrame
input_dict = {
    "Hours_Studied": hours_studied,
    "Access_to_Resources": access_to_resources,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Learning_Disabilities": learning_disabilities,
    "Parental_Involvement": parental_involvement,
    "Extracurricular_Activities": extracurricular_activities,
    "Previous_Scores": previous_scores,
    "Motivation_Level": motivation_level,
    "Internet_Access": internet_access,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Physical_Activity": physical_activity,
    "Parental_Education_Level": parental_education_level,
    "Distance_from_Home": distance_from_home,
    "Gender": gender
}

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Performance"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        performance_class = clf_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")
        st.info(f"Predicted Performance Class: {performance_class}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
