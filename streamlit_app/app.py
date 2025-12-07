import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------
# Load data and models
# ----------------------------
df = pd.read_csv("models/StudentPerformanceFactors_Cleaned.csv")

# Models
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

# Scalers & encoder
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
le = joblib.load("models/label_encoder.pkl")

# Columns used during training
all_training_columns = joblib.load("models/all_training_columns.pkl")

# ----------------------------
# App title
# ----------------------------
st.title("Student Performance Prediction")

# ----------------------------
# Main Features
# ----------------------------
st.header("Main Features")
hours_studied = st.number_input(
    "Hours Studied (per week)", value=int(df['Hours_Studied'].median())
)
attendance = st.number_input(
    "Attendance (%)", value=int(df['Attendance'].median())
)
parental_involvement = st.selectbox(
    "Parental Involvement", ["Low", "Medium", "High"], index=1
)
access_to_resources = st.selectbox(
    "Access to Resources", ["Low", "Medium", "High"], index=2
)
extracurricular_activities = st.selectbox(
    "Extracurricular Activities", ["No", "Yes"]
)
sleep_hours = st.number_input(
    "Sleep Hours (per day)", value=int(df['Sleep_Hours'].median())
)

# ----------------------------
# Optional Features (collapsed)
# ----------------------------
with st.expander("Optional Features"):
    previous_scores = st.number_input(
        "Previous Scores (%)", value=int(df['Previous_Scores'].median())
    )
    motivation_level = st.selectbox(
        "Motivation Level", ["Low", "Medium", "High"], index=1
    )
    internet_access = st.selectbox(
        "Internet Access", ["No", "Yes"]
    )
    tutoring_sessions = st.number_input(
        "Tutoring Sessions (per week)", value=int(df['Tutoring_Sessions'].median())
    )
    family_income = st.selectbox(
        "Family Income Level", ["Low", "Medium", "High"], index=1
    )
    teacher_quality = st.selectbox(
        "Teacher Quality", ["Low", "Medium", "High"], index=1
    )
    school_type = st.selectbox(
        "School Type", ["Public", "Private"]
    )
    peer_influence = st.selectbox(
        "Peer Influence", ["Negative", "Neutral", "Positive"], index=1
    )
    physical_activity = st.number_input(
        "Physical Activity (sessions per week)", value=int(df['Physical_Activity'].median())
    )
    learning_disabilities = st.selectbox(
        "Learning Disabilities", ["No", "Yes"]
    )
    parental_education_level = st.selectbox(
        "Parental Education Level", ["High School", "College", "Postgraduate"], index=1
    )
    distance_from_home = st.selectbox(
        "Distance from Home", ["Near", "Moderate", "Far"], index=1
    )
    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

# ----------------------------
# Predict button
# ----------------------------
if st.button("Predict"):
    # Collect input into DataFrame
    input_dict = {
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
        "Gender": [gender],
    }
    input_df = pd.DataFrame(input_dict)

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df)

    # Align columns to training
    for col in all_training_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[all_training_columns]

    # Scale features
    input_scaled_reg = scaler_reg.transform(input_encoded)
    input_scaled_clf = scaler_clf.transform(input_encoded)

    # Predict
    pred_score = reg_model.predict(input_scaled_reg)[0]
    pred_class = clf_model.predict(input_scaled_clf)[0]
    pred_class_label = le.inverse_transform([pred_class])[0]

    # Clip score to 0-100
    pred_score = np.clip(pred_score, 0, 100)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"Predicted Exam Score: {pred_score:.2f}")
    st.write(f"Predicted Performance Class: {pred_class_label}")
