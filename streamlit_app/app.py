import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load models and training column lists
# ==============================
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")

# Load the list of columns your models were trained on
all_lr_training_columns = joblib.load("models/all_lr_training_columns.pkl")  # LinearRegression
all_cb_training_columns = joblib.load("models/all_cb_training_columns.pkl")  # CatBoost

# ==============================
# Load CSV to reference features
# ==============================
train_df = pd.read_csv("data/StudentPerformanceFactors_Cleaned.csv")

target_col = "Exam_Score"
all_features = [c for c in train_df.columns if c != target_col]

categorical_features = [
    'Access_to_Resources', 'Parental_Involvement', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

numeric_features = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions',
    'Previous_Scores', 'Physical_Activity'
]

# Main 6 features
main_features = [
    "Hours_Studied", "Access_to_Resources", "Attendance",
    "Sleep_Hours", "Tutoring_Sessions", "Learning_Disabilities"
]

optional_features = [f for f in all_features if f not in main_features]

# ==============================
# Streamlit UI
# ==============================
st.title("Student Performance Prediction")
st.write("""
Predict student's exam score and performance classification.
Main 6 features are required; optional features are under "More Factors".
""")

# ------------------------------
# Main 6 inputs
# ------------------------------
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
access_to_resources = st.selectbox("Access to Resources", train_df["Access_to_Resources"].unique())
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, value=0)
learning_disabilities = st.selectbox("Learning Disabilities", train_df["Learning_Disabilities"].unique())

# ------------------------------
# Optional features in expander
# ------------------------------
optional_inputs = {}
with st.expander("More Factors (Optional)"):
    for col in optional_features:
        if col in categorical_features:
            optional_inputs[col] = st.selectbox(col, train_df[col].unique())
        else:
            optional_inputs[col] = st.number_input(col, value=int(train_df[col].median()))

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

# ==============================
# Prepare input for LinearRegression (one-hot encode categorical features)
# ==============================
X_lr = pd.get_dummies(input_df)
for col in all_lr_training_columns:
    if col not in X_lr.columns:
        X_lr[col] = 0  # missing columns filled with 0
X_lr = X_lr[all_lr_training_columns]  # reorder columns

# ==============================
# Prepare input for CatBoost (raw categorical features)
# ==============================
X_cb = input_df.copy()
for col in categorical_features:
    if col not in X_cb.columns:
        X_cb[col] = -1
X_cb = X_cb[all_cb_training_columns]  # reorder columns

# ==============================
# Prediction
# ==============================
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(X_lr)[0]
        st.success(f"Predicted Exam Score (Regression): {exam_score:.2f}")
    except Exception as e:
        st.error(f"Regression prediction failed: {e}")

    try:
        class_pred = clf_model.predict(X_cb)[0]
        st.success(f"Predicted Performance Class (Classification): {class_pred}")
    except Exception as e:
        st.error(f"Classification prediction failed: {e}")

