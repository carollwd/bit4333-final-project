import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Load models and preprocessing
# -------------------------------
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
le = joblib.load("models/label_encoder.pkl")
all_training_columns = joblib.load("models/all_training_columns.pkl")

# -------------------------------
# Convert numeric score to grade
# -------------------------------
def score_to_grade(score):
    if score >= 90:
        return 'A+'
    elif score >= 80:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 60:
        return 'C'
    else:
        return 'D'

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Student Performance Predictor")

st.write("### Main Features (Required)")
main_features = ["Hours_Studied", "Attendance", "Previous_Scores", "Motivation_Level",
                 "Teacher_Quality", "Parental_Involvement"]

main_inputs = {}
for col in main_features:
    if col in ["Motivation_Level", "Parental_Involvement", "Teacher_Quality"]:
        main_inputs[col] = st.selectbox(col, ["Low", "Medium", "High"])
    else:
        main_inputs[col] = st.number_input(col, value=0)

# Optional features in a collapsible section
with st.expander("Optional Features"):
    optional_features = ["Sleep_Hours", "Physical_Activity", "Internet_Access",
                         "Access_to_Resources", "Distance_from_Home",
                         "Extracurricular_Activities", "Family_Income",
                         "Gender"]
    optional_inputs = {}
    for col in optional_features:
        if col in ["Internet_Access", "Extracurricular_Activities", "Gender"]:
            optional_inputs[col] = st.selectbox(col, ["Yes", "No"] if col != "Gender" else ["Male", "Female"])
        elif col in ["Access_to_Resources", "Distance_from_Home", "Family_Income"]:
            optional_inputs[col] = st.selectbox(col, ["Low", "Medium", "High"] if col != "Distance_from_Home" else ["Near", "Moderate", "Far"])
        else:
            optional_inputs[col] = st.number_input(col, value=0)

# -------------------------------
# Combine inputs into DataFrame
# -------------------------------
input_dict = {**main_inputs, **optional_inputs}
input_df = pd.DataFrame([input_dict])

# Align features with training columns (one-hot encoding)
input_df_encoded = pd.get_dummies(input_df)
for col in all_training_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[all_training_columns]

# Scale features
input_df_scaled_reg = scaler_reg.transform(input_df_encoded)
input_df_scaled_clf = scaler_clf.transform(input_df_encoded)

# -------------------------------
# Make predictions
# -------------------------------
pred_exam_score = reg_model.predict(input_df_scaled_reg)[0]
pred_grade_from_score = score_to_grade(pred_exam_score)

pred_performance_class_enc = clf_model.predict(input_df_scaled_clf)[0]
pred_performance_class = le.inverse_transform([pred_performance_class_enc])[0]

# -------------------------------
# Display results
# -------------------------------
st.write(f"### Predicted Exam Score: {pred_exam_score:.2f}")
st.write(f"### Predicted Grade: {pred_grade_from_score}")
st.write(f"### Predicted Performance Class (Classification Model): {pred_performance_class}")
