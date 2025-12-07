# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder

st.title("Student Performance Prediction")

# Load models

reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

# Define features

main_features = [
    "Hours_Studied",
    "Attendance",
    "Access_to_Resources",
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Learning_Disabilities"
]

more_features = [
    "Parental_Involvement",
    "Extracurricular_Activities",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

all_features = main_features + more_features

# Input fields - Main factors

st.header("Main Factors (Required)")

hours_studied = st.number_input("Hours Studied per day", min_value=0, max_value=24, value=2)
attendance = st.selectbox("Attendance", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
sleep_hours = st.number_input("Sleep Hours per day", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions per week", min_value=0, max_value=10, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])

# Input fields - More factors (Optional)

st.header("More Factors (Optional)")

parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"], index=0)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=0)
motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
internet_access = st.selectbox("Internet Access", ["No", "Yes"], index=1)
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
school_type = st.selectbox("School Type", ["Public", "Private"], index=0)
peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=1)
physical_activity = st.number_input("Physical Activity Hours per week", min_value=0, max_value=50, value=3)
parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=0)
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], index=0)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)

# Prepare input DataFrame

input_dict = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Access_to_Resources": access_to_resources,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Learning_Disabilities": learning_disabilities,
    "Parental_Involvement": parental_involvement,
    "Extracurricular_Activities": extracurricular,
    "Previous_Scores": previous_scores,
    "Motivation_Level": motivation,
    "Internet_Access": internet_access,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Physical_Activity": physical_activity,
    "Parental_Education_Level": parental_education,
    "Distance_from_Home": distance_from_home,
    "Gender": gender
}

input_df = pd.DataFrame([input_dict])

# Encode categorical variables

cat_features = [
    "Attendance",
    "Access_to_Resources",
    "Learning_Disabilities",
    "Parental_Involvement",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

encoder = OrdinalEncoder()
input_df[cat_features] = encoder.fit_transform(input_df[cat_features])

# Predict

if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")

        class_pred = clf_model.predict(input_df)[0]
        st.info(f"Predicted Grade: {class_pred}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
