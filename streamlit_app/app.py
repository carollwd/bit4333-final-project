import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load models
# -------------------------------
reg_model_file = "models/best_reg_model.pkl"
clf_model_file = "models/best_clf_model.pkl"

reg_model = joblib.load(reg_model_file)
clf_model = joblib.load(clf_model_file)

# -------------------------------
# Load training data for defaults
# -------------------------------
train_df = pd.read_csv("data/StudentPerformanceFactors_Cleaned.csv")

# -------------------------------
# Helper functions
# -------------------------------
def map_categorical(value, mapping):
    return mapping.get(value, 0)

# Mappings
yes_no_map = {"Yes": 1, "No": 0}
level_map = {"Low": 0, "Medium": 1, "High": 2}
gender_map = {"Male": 0, "Female": 1}
school_map = {"Public":0, "Private":1}
peer_map = {"Positive":2, "Neutral":1, "Negative":0}
edu_map = {"High School":0, "College":1, "Postgraduate":2}
distance_map = {"Near":0, "Moderate":1, "Far":2}

# -------------------------------
# Features
# -------------------------------
main_features = [
    "Hours_Studied", "Attendance", "Access_to_Resources",
    "Sleep_Hours", "Tutoring_Sessions", "Learning_Disabilities"
]

all_features = reg_model.feature_names_in_

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Student Performance Predictor")
st.markdown("Predict Exam Score and Performance Level based on student factors.")

st.header("Main Factors (Required)")
hours_studied = st.number_input("Hours Studied per Day", 0, 24, 5)
attendance = st.number_input("Attendance (%)", 0, 100, 80)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
sleep_hours = st.number_input("Sleep Hours per Day", 0, 24, 7)
tutoring_sessions = st.number_input("Tutoring Sessions per Week", 0, 10, 0)
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])

# -------------------------------
# Optional factors (compressed)
# -------------------------------
st.markdown("---")
with st.expander("More Factors (Optional)"):
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"], index=1)
    previous_scores = st.number_input("Previous Scores", 0, 100, 70)
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
    internet_access = st.selectbox("Internet Access", ["Yes", "No"], index=1)
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
    school_type = st.selectbox("School Type", ["Public", "Private"], index=0)
    peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"], index=1)
    physical_activity = st.number_input("Physical Activity Hours/Week", 0, 20, 3)
    parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=1)
    distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], index=1)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)

# -------------------------------
# Prepare input with defaults
# -------------------------------
input_dict = {}

# Fill main factors
input_dict["Hours_Studied"] = hours_studied
input_dict["Attendance"] = attendance
input_dict["Access_to_Resources"] = map_categorical(access_to_resources, level_map)
input_dict["Sleep_Hours"] = sleep_hours
input_dict["Tutoring_Sessions"] = tutoring_sessions
input_dict["Learning_Disabilities"] = map_categorical(learning_disabilities, yes_no_map)

# Fill optional factors with defaults from training median
for feature in all_features:
    if feature not in input_dict:
        if feature == "Parental_Involvement":
            input_dict[feature] = map_categorical(parental_involvement, level_map)
        elif feature == "Extracurricular_Activities":
            input_dict[feature] = map_categorical(extracurricular_activities, yes_no_map)
        elif feature == "Previous_Scores":
            input_dict[feature] = previous_scores
        elif feature == "Motivation_Level":
            input_dict[feature] = map_categorical(motivation_level, level_map)
        elif feature == "Internet_Access":
            input_dict[feature] = map_categorical(internet_access, yes_no_map)
        elif feature == "Family_Income":
            input_dict[feature] = map_categorical(family_income, level_map)
        elif feature == "Teacher_Quality":
            input_dict[feature] = map_categorical(teacher_quality, level_map)
        elif feature == "School_Type":
            input_dict[feature] = map_categorical(school_type, school_map)
        elif feature == "Peer_Influence":
            input_dict[feature] = map_categorical(peer_influence, peer_map)
        elif feature == "Physical_Activity":
            input_dict[feature] = physical_activity
        elif feature == "Parental_Education_Level":
            input_dict[feature] = map_categorical(parental_education_level, edu_map)
        elif feature == "Distance_from_Home":
            input_dict[feature] = map_categorical(distance_from_home, distance_map)
        elif feature == "Gender":
            input_dict[feature] = map_categorical(gender, gender_map)
        else:
            # Fill any remaining optional features with median
            if feature in train_df.columns:
                input_dict[feature] = train_df[feature].median()
            else:
                input_dict[feature] = 0

input_df = pd.DataFrame([input_dict], columns=all_features)

# -------------------------------
# Predict
# -------------------------------
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        st.success(f"Predicted Exam Score: {exam_score:.2f}")

        performance_class = clf_model.predict(input_df)[0]
        st.info(f"Predicted Performance Level: {performance_class}")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
