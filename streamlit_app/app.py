import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------
# Load models
# ------------------------------
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

st.title("Student Performance Predictor")

st.markdown("### Enter student details below:")

# ------------------------------
# Main 6 features
# ------------------------------
hours_studied = st.number_input("Hours Studied per day", min_value=0, max_value=24, value=2)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
sleep_hours = st.number_input("Sleep Hours per day", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions per week", min_value=0, max_value=10, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

# ------------------------------
# Optional "More Factors" (collapsed)
# ------------------------------
with st.expander("More Factors (Optional)"):
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    internet_access = st.selectbox("Internet Access", ["No", "Yes"])
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
    physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0, max_value=40, value=3)
    parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# ------------------------------
# Map categorical values to numbers
# ------------------------------
def map_categorical(val, mapping):
    return mapping.get(val, 0)

# Example mappings
access_map = {"Low": 0, "Medium": 1, "High": 2}
yes_no_map = {"No": 0, "Yes": 1}
parent_invol_map = {"Low":0,"Medium":1,"High":2}
internet_map = {"No":0,"Yes":1}
income_map = {"Low":0,"Medium":1,"High":2}
teacher_map = {"Low":0,"Medium":1,"High":2}
school_map = {"Public":0,"Private":1}
peer_map = {"Negative":0,"Neutral":1,"Positive":2}
distance_map = {"Near":0,"Moderate":1,"Far":2}
gender_map = {"Male":0,"Female":1}
learning_map = {"No":0,"Yes":1}
parent_edu_map = {"High School":0,"College":1,"Postgraduate":2}

# ------------------------------
# Build input dataframe
# ------------------------------
input_dict = {
    "Hours_Studied": [hours_studied],
    "Access_to_Resources": [map_categorical(access_to_resources, access_map)],
    "Attendance": [attendance],
    "Sleep_Hours": [sleep_hours],
    "Tutoring_Sessions": [tutoring_sessions],
    "Learning_Disabilities": [map_categorical(learning_disabilities, learning_map)],
}

# Add optional features if user provided them
if 'parental_involvement' in locals():
    input_dict.update({
        "Parental_Involvement": [map_categorical(parental_involvement, parent_invol_map)],
        "Internet_Access": [map_categorical(internet_access, internet_map)],
        "Family_Income": [map_categorical(family_income, income_map)],
        "Teacher_Quality": [map_categorical(teacher_quality, teacher_map)],
        "School_Type": [map_categorical(school_type, school_map)],
        "Peer_Influence": [map_categorical(peer_influence, peer_map)],
        "Physical_Activity": [physical_activity],
        "Parental_Education_Level": [map_categorical(parental_education, parent_edu_map)],
        "Distance_from_Home": [map_categorical(distance_from_home, distance_map)],
        "Gender": [map_categorical(gender, gender_map)]
    })

input_df = pd.DataFrame(input_dict)

# ------------------------------
# Predict
# ------------------------------
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        grade = clf_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")
        st.success(f"Predicted Grade: {grade}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
