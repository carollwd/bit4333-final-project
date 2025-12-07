import streamlit as st
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("models/StudentPerformanceFactors_Cleaned.csv")

# Remove target columns – not used as inputs
TARGET_COLS = ["Exam_Score", "GradeCategory"]
df_input = df.drop(columns=TARGET_COLS)

# Detect numerical + categorical columns
numeric_cols = df_input.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df_input.select_dtypes(include=["object"]).columns.tolist()

# Load saved preprocessing + models + training columns
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
training_columns = joblib.load("models/all_training_columns.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

st.title("Student Performance Predictor")

st.write("Enter the student details below:")

# Store user inputs
user_input = {}

# Numeric fields
for col in numeric_cols:
    user_input[col] = st.number_input(
        col, 
        value=float(df[col].median())
    )

# Categorical dropdowns
for col in categorical_cols:
    user_input[col] = st.selectbox(
        col,
        df[col].unique().tolist()
    )

if st.button("Predict"):
    try:
        # Convert inputs to df
        input_df = pd.DataFrame([user_input])

        # Apply one-hot encoding
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Force alignment with training columns
        for col in training_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[training_columns]

        # Run predictions
        reg_scaled = scaler_reg.transform(input_encoded)
        clf_scaled = scaler_clf.transform(input_encoded)

        exam_score = reg_model.predict(reg_scaled)[0]
        grade_pred = clf_model.predict(clf_scaled)[0]
        final_grade = label_encoder.inverse_transform([grade_pred])[0]

        st.success(f"Predicted Exam Score: {exam_score:.2f}%")
        st.success(f"Predicted Grade Category: {final_grade}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
