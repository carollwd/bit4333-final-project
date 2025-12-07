# app.py
import streamlit as st
import pandas as pd
import joblib

# Load models
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

# Load training data for defaults
train_df = pd.read_csv("data/StudentPerformanceFactors_Cleaned.csv")

# App layout
st.title("Student Performance Predictor")
st.write("Enter the student's details to predict their exam score and grade.")

# Main 6 inputs
st.header("Main Factors (required)")
hours_studied = st.number_input("Hours Studied (per day)", min_value=0, max_value=24, value=2)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
sleep_hours = st.number_input("Sleep Hours (per day)", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions (per week)", min_value=0, max_value=7, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

# Optional factors
st.header("More Factors (optional)")
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.selectbox("Internet Access", ["No", "Yes"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0, max_value=20, value=3)
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)

# Create input DataFrame
input_dict = {
    "Hours_Studied": hours_studied,
    "Access_to_Resources": access_to_resources,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Learning_Disabilities": learning_disabilities,
    # Optional factors
    "Parental_Involvement": parental_involvement,
    "Motivation_Level": motivation_level,
    "Internet_Access": internet_access,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Physical_Activity": physical_activity,
    "Parental_Education_Level": parental_education_level,
    "Distance_from_Home": distance_from_home,
    "Gender": gender,
    "Previous_Scores": previous_scores
}

input_df = pd.DataFrame([input_dict])

# Fill missing features with dataset defaults
all_features = reg_model.feature_names_in_  # features used by model
for col in all_features:
    if col not in input_df.columns:
        if train_df[col].dtype.kind in 'iufc':  # numeric
            input_df[col] = train_df[col].mean()
        else:  # categorical
            input_df[col] = train_df[col].mode()[0]

# Predict
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        grade = clf_model.predict(input_df)[0]

        st.subheader("Prediction Results")
        st.write(f"Predicted Exam Score: {exam_score:.2f}")
        st.write(f"Predicted Grade: {grade}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
