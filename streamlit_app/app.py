# streamlit_app/app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# 1. Load models, scaler, encoder
# -----------------------------
reg_model = joblib.load("models/best_reg_model.pkl")
clf_model = joblib.load("models/best_clf_model.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")
scaler_clf = joblib.load("models/scaler_clf.pkl")
le = joblib.load("models/label_encoder.pkl")
all_cols = joblib.load("models/all_training_columns.pkl")  # full feature list

# -----------------------------
# 2. Define features
# -----------------------------
main_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 
                 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']

optional_features = ['Parental_Involvement', 'Access_to_Resources',
                     'Extracurricular_Activities', 'Motivation_Level',
                     'Internet_Access', 'Family_Income', 'Teacher_Quality',
                     'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                     'Parental_Education_Level', 'Distance_from_Home', 'Gender']

# -----------------------------
# 3. User input
# -----------------------------
st.title("Student Performance Predictor")

st.subheader("Main Features (Required)")
main_inputs = {}
for col in main_features:
    main_inputs[col] = st.number_input(col, value=0)

st.subheader("Optional Features (can skip)")
optional_inputs = {}
for col in optional_features:
    val = 0
    if col in ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
               'Tutoring_Sessions', 'Physical_Activity']:
        val = st.number_input(col, value=0)
    else:
        val = st.selectbox(col, options=['Low','Medium','High','Yes','No','Male','Female'])
    optional_inputs[col] = val

# -----------------------------
# 4. Build dataframe
# -----------------------------
data = {**main_inputs, **optional_inputs}
df_input = pd.DataFrame([data])

# -----------------------------
# 5. Encode categorical
# -----------------------------
cat_cols = df_input.select_dtypes(include='object').columns
df_input_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)

# -----------------------------
# 6. Align with training columns
# -----------------------------
for col in all_cols:
    if col not in df_input_encoded.columns:
        df_input_encoded[col] = 0  # add missing columns

df_input_encoded = df_input_encoded[all_cols]  # reorder

# -----------------------------
# 7. Scale features
# -----------------------------
df_input_scaled_reg = scaler_reg.transform(df_input_encoded)
df_input_scaled_clf = scaler_clf.transform(df_input_encoded)

# -----------------------------
# 8. Predictions
# -----------------------------
pred_reg = reg_model.predict(df_input_scaled_reg)[0]
pred_clf = clf_model.predict(df_input_scaled_clf)[0]
pred_clf_label = le.inverse_transform([pred_clf])[0]

# -----------------------------
# 9. Show results
# -----------------------------
st.subheader("Predictions")
st.write(f"Predicted Exam Score: {pred_reg:.2f} / 100")
st.write(f"Predicted Performance Grade: {pred_clf_label}")
