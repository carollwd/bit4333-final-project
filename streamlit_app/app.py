# ==============================
# Training models and saving columns for Streamlit
# ==============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from catboost import CatBoostClassifier
import joblib

# Load cleaned dataset
df = pd.read_csv("data/StudentPerformanceFactors_Cleaned.csv")

# Separate target
target_col = "Exam_Score"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical and numeric features
categorical_features = [
    'Access_to_Resources', 'Parental_Involvement', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

numeric_features = [c for c in X.columns if c not in categorical_features]

# ------------------------------
# Prepare data for LinearRegression (one-hot encode)
# ------------------------------
X_lr = pd.get_dummies(X, columns=categorical_features)
y_lr = y.copy()

# Train-test split
X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

# Train LinearRegression
reg_model = LinearRegression()
reg_model.fit(X_lr_train, y_lr_train)

# Save model
joblib.dump(reg_model, "models/best_reg_model.pkl")

# Save the columns for Streamlit
joblib.dump(X_lr_train.columns.tolist(), "models/all_lr_training_columns.pkl")

# ------------------------------
# Prepare data for CatBoost (raw categorical features)
# ------------------------------
X_cb = X.copy()
y_cb = pd.cut(y, bins=[0, 59, 69, 79, 100], labels=["F", "D", "C", "B"])  # Example grading

# Train-test split
X_cb_train, X_cb_test, y_cb_train, y_cb_test = train_test_split(X_cb, y_cb, test_size=0.2, random_state=42)

# Train CatBoostClassifier
clf_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=4,
    verbose=0
)
clf_model.fit(X_cb_train, y_cb_train, cat_features=categorical_features)

# Save model
joblib.dump(clf_model, "models/best_clf_model.pkl")

# Save columns for Streamlit
joblib.dump(X_cb_train.columns.tolist(), "models/all_cb_training_columns.pkl")

print("✅ Models and column lists saved successfully!")
