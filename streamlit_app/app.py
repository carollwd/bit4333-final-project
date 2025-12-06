import streamlit as st
import pandas as pd
import joblib

# Load models

reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

# App title

st.title("Student Performance Prediction")

# User inputs

hours_studied = st.number_input("Hours Studied", min_value=0, max_value=100, value=20)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=0)
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=20, value=3)
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Make DataFrame

input_df = pd.DataFrame({
    "Hours_Studied": [hours_studied],
    "Attendance": [attendance],
    "Parental_Involvement": [parental_involvement],
    "Access_to_Resources": [access_to_resources],
    "Extracurricular_Activities": [extracurricular_activities],
    "Sleep_Hours": [sleep_hours],
    "Previous_Scores": [previous_scores],
    "Motivation_Level": [motivation_level],
    "Internet_Access": [internet_access],
    "Tutoring_Sessions": [tutoring_sessions],
    "Family_Income": [family_income],
    "Teacher_Quality": [teacher_quality],
    "School_Type": [school_type],
    "Peer_Influence": [peer_influence],
    "Physical_Activity": [physical_activity],
    "Learning_Disabilities": [learning_disabilities],
    "Parental_Education_Level": [parental_education_level],
    "Distance_from_Home": [distance_from_home],
    "Gender": [gender]
})

# Predict button

if st.button("Predict"):
    # Regression prediction
    exam_score = reg_model.predict(input_df)[0]
    st.write(f"Predicted Exam Score: {exam_score:.2f}")
    
    # Classification prediction
    performance_class = clf_model.predict(input_df)[0]
    st.write(f"Predicted Performance: {performance_class}")
