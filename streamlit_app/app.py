# Cell 1: Imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load models & helpers
# ==============================
# Make sure these files are in the "models" folder or uploaded
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
le = joblib.load("models/label_encoder.pkl")
all_features = joblib.load("models/all_features.pkl")  # list of all training features

# Load CSV to get column names & categorical info
df = pd.read_csv("models/StudentPerformanceFactors_Cleaned.csv")
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# ==============================
# Streamlit UI
# ==============================
st.title("Student Performance Prediction")
st.write("Predict student's exam score (0-100) and grade category.\n"
         "Fill the main 6 features first, optional features can improve accuracy.")

# ------------------------------
# Main 6 inputs
# ------------------------------
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

# ------------------------------
# Optional features in collapsible section
# ------------------------------
optional_inputs = {}
with st.expander("More Factors (Optional)"):
    for col in all_features:
        if col.lower().replace("_","") in ["hoursstudied","accesstoresources",
                                           "attendance","sleephours",
                                           "tutoringsessions","learningdisabilities"]:
            continue  # skip main 6
        if col in categorical_cols:
            optional_inputs[col] = st.selectbox(col, df[col].unique())
        else:
            optional_inputs[col] = st.number_input(col, value=int(df[col].median()))

# ------------------------------
# Prepare input DataFrame
# ------------------------------
input_data = {
    "Hours_Studied": hours_studied,
    "Access_to_Resources": access_to_resources,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Tutoring_Sessions": tutoring_sessions,
    "Learning_Disabilities": learning_disabilities
}
input_data.update(optional_inputs)
input_df = pd.DataFrame([input_data])

# Encode categorical features exactly like training
for col in categorical_cols:
    if col in input_df.columns:
        input_df[col] = pd.Categorical(input_df[col], categories=df[col].unique()).codes

# Add missing columns
for col in all_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[all_features]

# Scale features
X_reg_scaled = scaler_reg.transform(input_df)
X_clf_scaled = scaler_clf.transform(input_df)

# ==============================
# Prediction
# ==============================
if st.button("Predict"):
    # Regression: Exam Score
    try:
        exam_score = reg_model.predict(X_reg_scaled)[0]
        exam_score = max(0, min(100, exam_score))  # clip 0-100
        st.success(f"Predicted Exam Score: {exam_score:.2f}")
    except Exception as e:
        st.error(f"Regression prediction failed: {e}")

    # Classification: Grade
    try:
        grade_enc = clf_model.predict(X_clf_scaled)[0]
        grade = le.inverse_transform([grade_enc])[0]
        st.success(f"Predicted Grade: {grade}")
    except Exception as e:
        st.error(f"Classification prediction failed: {e}")
