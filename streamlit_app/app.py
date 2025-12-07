import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================
# Load models and scaler/encoder
# ==============================
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
le = joblib.load("models/label_encoder.pkl")  # for GradeCategory

# ==============================
# Load CSV to get features
# ==============================
df = pd.read_csv("models/StudentPerformanceFactors_Cleaned.csv")
target_col = "Exam_Score"
all_features = [c for c in df.columns if c != target_col]

# Categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# ==============================
# Streamlit UI
# ==============================
st.title("Student Performance Prediction")
st.write("Predict student's exam score and grade. Main 6 features required, others optional.")

# ----------------------------
# Main 6 inputs
# ----------------------------
hours_studied = st.number_input("Hours Studied", 0, 24, 5)
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
attendance = st.slider("Attendance (%)", 0, 100, 75)
sleep_hours = st.number_input("Sleep Hours", 0, 24, 7)
tutoring_sessions = st.number_input("Tutoring Sessions", 0, 10, 0)
learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])

# ----------------------------
# Optional features (collapsed)
# ----------------------------
optional_inputs = {}
with st.expander("More Factors (Optional)"):
    for col in all_features:
        if col.lower().replace("_","") in ["hoursstudied","accesstoresources",
                                           "attendance","sleephours",
                                           "tutoringsessions","learningdisabilities"]:
            continue
        if col in df.columns:
            if col in categorical_cols:
                optional_inputs[col] = st.selectbox(col, df[col].unique())
            else:
                optional_inputs[col] = st.number_input(col, value=int(df[col].median()))
        else:
            optional_inputs[col] = 0  # default if column missing

# ----------------------------
# Prepare input dataframe
# ----------------------------
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

# Encode categorical features
for col in categorical_cols:
    if col in input_df.columns:
        input_df[col] = pd.Categorical(input_df[col],
                                       categories=df[col].unique()).codes

# Ensure all features are present and in correct order
missing_cols = set(all_features) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[all_features]

# Scale features
input_df_scaled_reg = scaler_reg.transform(input_df)
input_df_scaled_clf = scaler_clf.transform(input_df)

# ==============================
# Prediction
# ==============================
if st.button("Predict"):
    try:
        exam_score = reg_model.predict(input_df_scaled_reg)[0]
        exam_score = max(0, min(100, exam_score))  # ensure within 0-100%
        st.success(f"Predicted Exam Score: {exam_score:.2f}")
    except Exception as e:
        st.error(f"Regression failed: {e}")

    try:
        class_pred = clf_model.predict(input_df_scaled_clf)[0]
        grade = le.inverse_transform([class_pred])[0]
        st.success(f"Predicted Grade: {grade}")
    except Exception as e:
        st.error(f"Classification failed: {e}")
