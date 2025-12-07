import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models, scalers, encoder, and training columns
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
le = joblib.load("models/label_encoder.pkl")
all_cols = joblib.load("models/all_training_columns.pkl")

# Main and optional features
main_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 
                 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']

optional_features_options = {
    'Parental_Involvement': ['Low','Medium','High'],
    'Access_to_Resources': ['Low','Medium','High'],
    'Extracurricular_Activities': ['No','Yes'],
    'Motivation_Level': ['Low','Medium','High'],
    'Internet_Access': ['No','Yes'],
    'Family_Income': ['Low','Medium','High'],
    'Teacher_Quality': ['Low','Medium','High'],
    'School_Type': ['Public','Private'],
    'Peer_Influence': ['Negative','Neutral','Positive'],
    'Learning_Disabilities': ['No','Yes'],
    'Parental_Education_Level': ['High School','College','Postgraduate'],
    'Distance_from_Home': ['Near','Moderate','Far'],
    'Gender': ['Male','Female']
}

# Streamlit UI
st.title("Student Performance Predictor")
st.subheader("Main Features (Required)")

# User inputs for main features
main_inputs = {col: st.number_input(col, value=0) for col in main_features}

# Optional features in collapsible expander
optional_inputs = {}
with st.expander("Optional Features (Click to Expand)"):
    for col, options in optional_features_options.items():
        optional_inputs[col] = st.selectbox(col, options)

# Predict button
if st.button("Predict"):
    # Build dataframe
    df_input = pd.DataFrame([{**main_inputs, **optional_inputs}])

    # Encode categorical features
    cat_cols = df_input.select_dtypes(include='object').columns
    df_input_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

    # Align columns
    for col in all_cols:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0
    df_input_encoded = df_input_encoded[all_cols]

    # Scale
    df_input_scaled_reg = scaler_reg.transform(df_input_encoded)
    df_input_scaled_clf = scaler_clf.transform(df_input_encoded)

    # Predict
    pred_reg = reg_model.predict(df_input_scaled_reg)[0]
    pred_clf_label = le.inverse_transform([clf_model.predict(df_input_scaled_clf)[0]])[0]

    # Show results
    st.subheader("Predictions")
    st.write(f"Predicted Exam Score: {pred_reg:.2f} / 100")
    st.write(f"Predicted Performance Grade: {pred_clf_label}")
