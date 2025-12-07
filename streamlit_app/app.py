import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ==============================
# Load models
# ==============================
reg_model_file = "models/best_reg_model.pkl"
clf_model_file = "models/best_clf_model.pkl"

reg_model = joblib.load(reg_model_file)
clf_model = joblib.load(clf_model_file)

# ==============================
# Define all features manually
# Must be in same order as used during training
# ==============================
all_features = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
    # Add the remaining features exactly as in training dataset
    "Feature20", "Feature21", "Feature22", "Feature23", "Feature24", "Feature25", "Feature26"
]

# ==============================
# Streamlit UI
# ==============================
st.title("Student Performance Predictor")
st.write("Predict student exam score and grade based on performance factors.")

st.header("Main Factors (Required)")
hours_studied = st.number_input("Hours Studied per Day", min_value=0, max_value=24, value=2)
attendance = st.selectbox("Attendance", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions per Week", min_value=0, max_value=20, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

st.header("Additional Factors (Optional)")
previous_scores = st.number_input("Previous Scores (Average)", min_value=0, max_value=100, value=70)
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
internet_access = st.selectbox("Internet Access", ["No", "Yes"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=50, value=3)
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Set default values for remaining features
feature_defaults = [0]*(len(all_features) - 19)

# ==============================
# Prepare input DataFrame
# ==============================
input_data = [
    hours_studied,
    attendance,
    access_to_resources,
    extracurricular_activities,
    sleep_hours,
    previous_scores,
    motivation_level,
    internet_access,
    tutoring_sessions,
    family_income,
    teacher_quality,
    school_type,
    peer_influence,
    physical_activity,
    learning_disabilities,
    parental_education_level,
    distance_from_home,
    gender
] + feature_defaults

input_df = pd.DataFrame([input_data], columns=all_features)

# ==============================
# Encode categorical features
# ==============================
for col in input_df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    # Fit on training categories manually or use previous mapping
    # Here, we fit on input only for simplicity
    input_df[col] = le.fit_transform(input_df[col])

# ==============================
# Predict
# ==============================
if st.button("Predict"):
    try:
        # Regression prediction
        exam_score = reg_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")

        # Classification prediction
        if clf_model:
            grade = clf_model.predict(input_df)[0]
            st.success(f"Predicted Grade: {grade}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
