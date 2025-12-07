import streamlit as st
import pandas as pd
import joblib

# Load models
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

st.title("Student Performance Predictor")
st.markdown("Predict a student's exam score and performance class based on main and optional factors.")

# Main 6 factors (always visible)
st.header("Main Factors (Required)")
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions per week", min_value=0, max_value=10, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

# Optional factors (collapsible)
with st.expander("More Factors (Optional)"):
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])
    previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=60)
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    internet_access = st.selectbox("Internet Access", ["No", "Yes"])
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
    physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=50, value=3)
    parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    
# Map strings to numbers
mapping_dict = {
    "Low": 0, "Medium": 1, "High": 2,
    "No": 0, "Yes": 1,
    "Negative": 0, "Neutral": 1, "Positive": 2,
    "Public": 0, "Private": 1,
    "High School": 0, "College": 1, "Postgraduate": 2,
    "Near": 0, "Moderate": 1, "Far": 2,
    "Male": 0, "Female": 1
}

# Prepare input dataframe
input_dict = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Access_to_Resources": mapping_dict[access_to_resources],
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Learning_Disabilities": mapping_dict[learning_disabilities],
    # Optional factors
    "Parental_Involvement": mapping_dict.get(parental_involvement, 1),
    "Extracurricular_Activities": mapping_dict.get(extracurricular_activities, 0),
    "Previous_Scores": previous_scores if 'previous_scores' in locals() else 60,
    "Motivation_Level": mapping_dict.get(motivation_level, 1),
    "Internet_Access": mapping_dict.get(internet_access, 1),
    "Family_Income": mapping_dict.get(family_income, 1),
    "Teacher_Quality": mapping_dict.get(teacher_quality, 1),
    "School_Type": mapping_dict.get(school_type, 0),
    "Peer_Influence": mapping_dict.get(peer_influence, 1),
    "Physical_Activity": physical_activity if 'physical_activity' in locals() else 3,
    "Parental_Education_Level": mapping_dict.get(parental_education_level, 1),
    "Distance_from_Home": mapping_dict.get(distance_from_home, 1),
    "Gender": mapping_dict.get(gender, 0)
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Performance"):
    try:
        exam_score = reg_model.predict(input_df)[0]
        class_pred = clf_model.predict(input_df)[0]

        st.success(f"Predicted Exam Score: {exam_score:.2f}")

        # Color-coded performance
        color = "black"
        if class_pred.lower() == "high":
            color = "green"
        elif class_pred.lower() == "medium":
            color = "orange"
        elif class_pred.lower() == "low":
            color = "red"

        st.markdown(f"**Predicted Performance Class:** <span style='color:{color}'>{class_pred}</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

