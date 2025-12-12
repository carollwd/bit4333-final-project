# Student Performance Predictor

## About Project
This project predicts student exam scores using machine learning regression models. The aim is to understand key factors influencing academic performance and provide a tool for exploring hypothetical scenarios.

## Background Overview
Student performance depends on multiple factors including study habits, attendance, parental involvement, and sleep. Traditional statistical analysis may miss complex relationships, whereas machine learning can model interactions and provide more accurate predictions.

## Problem Statement
Identifying the factors that most influence student exam scores is critical for academic planning and intervention. Accurate prediction can help simulate outcomes based on different student behaviors and circumstances.

## Objective
- Identify key features that affect student performance.  
- Train and evaluate regression models (Linear Regression, SVR, XGBoost, CatBoost).  
- Deploy an interactive Streamlit app for predicting exam scores using user-input features.  

## Built With
- Python  
- Pandas, NumPy, Scikit-learn, XGBoost, CatBoost  
- Streamlit (interactive web app)  

## Data Source
- Dataset: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)  
- Features include: Hours_Studied, Attendance, Motivation_Level, Parental_Involvement, Sleep_Hours, Previous_Scores, and other optional demographic and behavioral features.  

## Methodology
1. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numeric features.  
2. **Feature Selection**: Focus on main predictors: Hours_Studied, Attendance, Previous_Scores, Motivation_Level, Parental_Involvement, Sleep_Hours.  
3. **Model Development**: Train Linear Regression, SVR, XGBoost, and CatBoost regressors.  
4. **Model Evaluation**: Metrics: MSE and R²; best model: Linear Regression (MSE = 3.256, R² = 0.770).  
5. **Deployment**: Streamlit app allows interactive input for predicting exam scores based on primary and optional features.  

## Result and Impact
- Linear Regression model provides accurate predictions of exam scores.  
- Key contributing factors: **Attendance, Hours Studied, Previous Scores**.  
- Streamlit app demonstrates how changes in student behavior may impact exam outcomes.  

## How to Use
1. Open `streamlit_app/app.py`  
2. Upload the cleaned dataset (`StudentPerformanceFactors_Cleaned.csv`)  
3. Upload model files (`best_reg_model.pkl`, `scaler_reg.pkl`, `all_training_columns.pkl`)  
4. Input values for main and optional features  
5. Click **Predict** to see the estimated exam score  

## Acknowledgements
- Mr. Nazmirul Izzad Bin Nassir – guidance and support throughout the project.  
- Kaggle – source of the student performance dataset.  
