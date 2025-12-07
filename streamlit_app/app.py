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

# Define features
main_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 
                 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']
optional_features = ['Parental_Involvement', 'Access_to_Resources',
                     'Extracurricular_Activities', 'Motivation_Level',
                     'Internet_Access', 'Family_Income', 'Teacher_Quality',
                     'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                     'Parental_Education_Level', 'Distance_from_Home', 'Gender']

# User input
st.title("Student Performance Predictor")
st.subheader("Main Features (Required)")
main_inputs = {col: st.number_input(col, value=0) for col in main_features}

# Optional features inside collapsible expander
optional_inputs = {}
with st.expander("Optional Features (Click to Expand)"):
    for col in optional_features:
        # Decide input type based on expected values
        if col in ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities',
                   'Motivation_Level','Internet_Access','Family_Income','Teacher_Quality',
                   'School_Type','Peer_Influence','Learning_Disabilities',
                   'Parental_Education_Level','Distance_from_Home','Gender']:
            optional_inputs[col] = st.selectbox(col, options=['Low','Medium','High','Yes','No','Male','Female'])
        else:  # numeric inputs
            optional_inputs[col] = st.number_input(col, value=0)

# Build dataframe
df_input = pd.DataFrame([{**main_inputs, **optional_inputs}])

# Encode categorical
cat_cols = df_input.select_dtypes(include='object').columns
df_input_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

# Align with training columns
for col in all_cols:
    if col not in df_input_encoded.columns:
        df_input_encoded[col] = 0
df_input_encoded = df_input_encoded[all_cols]

# Scale features
df_input_scaled_reg = scaler_reg.transform(df_input_encoded)
df_input_scaled_clf = scaler_clf.transform(df_input_encoded)

# Predictions
pred_reg = reg_model.predict(df_input_scaled_reg)[0]
pred_clf_label = le.inverse_transform([clf_model.predict(df_input_scaled_clf)[0]])[0]

# Show results
st.subheader("Predictions")
st.write(f"Predicted Exam Score: {pred_reg:.2f} / 100")
st.write(f"Predicted Performance Grade: {pred_clf_label}")
