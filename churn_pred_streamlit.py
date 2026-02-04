import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
st.set_page_config(page_title="Account Churn Prediction", page_icon="📉")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "Churn.joblib")
try:
   scaler = joblib.load(SCALER_PATH)
except Exception as e:
   st.error(f"Could not load scaler from {SCALER_PATH}: {e}")
   st.stop()
try:
   model = joblib.load(MODEL_PATH)
except Exception as e:
   st.error(f"Could not load model from {MODEL_PATH}: {e}")
   st.stop()
st.title("Account Churn Prediction")
st.write("Enter Account Transaction details and click **Predict Churn**.")
st.divider()
# ----------- USER INPUTS -----------
AGE = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
TENURE = st.number_input("Tenure (in months)", min_value=0, max_value=600, value=12, step=1)
SEX = st.selectbox("Enter the gender", ["Male", "Female"])
gender_selected = 0 if SEX == "Male" else 1   # ✅ FIXED: use numeric encoding
CURRENTBALANCE = st.number_input("Current Balance", min_value=0.0, value=1000.0, step=100.0, format="%.2f")
PRODUCTS = st.number_input("Number of Products", min_value=1, max_value=10, value=2, step=1)
CREDIT_SCORE = st.number_input("Credit Score", min_value=1, max_value=100, value=50, step=1)
GROWTH_RATE = st.number_input("Growth Rate", min_value=-100.0, max_value=100.0, value=0.10, step=0.01, format="%.2f")
AVG_TRANS_VALUE = st.number_input("Average Transaction Value", min_value=0.0, value=100.0, step=10.0, format="%.2f")
DEBIT_TRANS_RATIO = st.number_input("Debit Transaction Ratio (0.01+)", min_value=0.0, max_value=10000000.0, value=0.50, step=0.01, format="%.2f")
TRANSACTION_FREQ = st.number_input("Transaction Frequency", min_value=0, max_value=100000, value=10, step=1)
DEBIT_INACTIVTY_MONTHS = st.number_input("Debit Inactivity Month(s)", min_value=0, max_value=600, value=10, step=1)
DEBIT_SLOPE_TREND = st.number_input("Debit Slope", min_value=0, max_value=100000, value=10, step=1)
st.divider()
FEATURE_COLUMNS = [
   'SEX',
   'AGE',
   'CURRENTBALANCE',
   'PRODUCTS',
   'CREDIT_SCORE',
   'TENURE',
   'GROWTH_RATE',
   'AVG_TRANS_VALUE',
   'DEBIT_TRANS_RATIO',
   'TRANSACTION_FREQ',
   'DEBIT_INACTIVTY_MONTHS',
   'DEBIT_SLOPE_TREND'
]
def build_feature_df():
   values = [
       gender_selected,   # ✅ FIXED: use 0/1 instead of "Male/Female"
       AGE,
       CURRENTBALANCE,
       PRODUCTS,
       CREDIT_SCORE,
       TENURE,
       GROWTH_RATE,
       AVG_TRANS_VALUE,
       DEBIT_TRANS_RATIO,
       TRANSACTION_FREQ,
       DEBIT_INACTIVTY_MONTHS,
       DEBIT_SLOPE_TREND
   ]
   df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
   # Ensure all numeric
   for col in df.columns:
       df[col] = pd.to_numeric(df[col], errors="coerce")
   return df
# ----------- PREDICTION -----------
if st.button("Predict Churn"):
   try:
       X_df = build_feature_df()
       # Helpful checks (won’t break app)
       if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != X_df.shape[1]:
           st.warning(
               f"Scaler expects {scaler.n_features_in_} features, "
               f"but you provided {X_df.shape[1]}. Check FEATURE_COLUMNS/order."
           )
       if hasattr(model, "n_features_in_") and model.n_features_in_ != X_df.shape[1]:
           st.warning(
               f"Model expects {model.n_features_in_} features, "
               f"but you provided {X_df.shape[1]}. Check FEATURE_COLUMNS/order."
           )
       X_scaled = scaler.transform(X_df)
       y_pred = model.predict(X_scaled)[0]
       label = "Churn" if int(y_pred) == 1 else "Not Churn"
       if hasattr(model, "predict_proba"):
           proba = model.predict_proba(X_scaled)[0][1]
           st.success(f"**Prediction:** {label}  |  **Churn probability:** {proba:.2%}")
       else:
           st.success(f"**Prediction:** {label}")
       with st.expander("Features sent to the model"):
           st.dataframe(X_df)
   except Exception as e:
       st.error(f"Prediction failed: {e}")import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
st.set_page_config(page_title="Account Churn Prediction", page_icon="📉")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "Churn.joblib")
try:
   scaler = joblib.load(SCALER_PATH)
except Exception as e:
   st.error(f"Could not load scaler from {SCALER_PATH}: {e}")
   st.stop()
try:
   model = joblib.load(MODEL_PATH)
except Exception as e:
   st.error(f"Could not load model from {MODEL_PATH}: {e}")
   st.stop()
st.title("Account Churn Prediction")
st.write("Enter Account Transaction details and click **Predict Churn**.")
st.divider()
# ----------- USER INPUTS -----------
AGE = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
TENURE = st.number_input("Tenure (in months)", min_value=0, max_value=600, value=12, step=1)
SEX = st.selectbox("Enter the gender", ["Male", "Female"])
gender_selected = 0 if SEX == "Male" else 1   # ✅ FIXED: use numeric encoding
CURRENTBALANCE = st.number_input("Current Balance", min_value=0.0, value=1000.0, step=100.0, format="%.2f")
PRODUCTS = st.number_input("Number of Products", min_value=1, max_value=10, value=2, step=1)
CREDIT_SCORE = st.number_input("Credit Score", min_value=1, max_value=100, value=50, step=1)
GROWTH_RATE = st.number_input("Growth Rate", min_value=-100.0, max_value=100.0, value=0.10, step=0.01, format="%.2f")
AVG_TRANS_VALUE = st.number_input("Average Transaction Value", min_value=0.0, value=100.0, step=10.0, format="%.2f")
DEBIT_TRANS_RATIO = st.number_input("Debit Transaction Ratio (0.01+)", min_value=0.0, max_value=10000000.0, value=0.50, step=0.01, format="%.2f")
TRANSACTION_FREQ = st.number_input("Transaction Frequency", min_value=0, max_value=100000, value=10, step=1)
DEBIT_INACTIVTY_MONTHS = st.number_input("Debit Inactivity Month(s)", min_value=0, max_value=600, value=10, step=1)
DEBIT_SLOPE_TREND = st.number_input("Debit Slope", min_value=0, max_value=100000, value=10, step=1)
st.divider()
FEATURE_COLUMNS = [
   'SEX',
   'AGE',
   'CURRENTBALANCE',
   'PRODUCTS',
   'CREDIT_SCORE',
   'TENURE',
   'GROWTH_RATE',
   'AVG_TRANS_VALUE',
   'DEBIT_TRANS_RATIO',
   'TRANSACTION_FREQ',
   'DEBIT_INACTIVTY_MONTHS',
   'DEBIT_SLOPE_TREND'
]
def build_feature_df():
   values = [
       gender_selected,   # ✅ FIXED: use 0/1 instead of "Male/Female"
       AGE,
       CURRENTBALANCE,
       PRODUCTS,
       CREDIT_SCORE,
       TENURE,
       GROWTH_RATE,
       AVG_TRANS_VALUE,
       DEBIT_TRANS_RATIO,
       TRANSACTION_FREQ,
       DEBIT_INACTIVTY_MONTHS,
       DEBIT_SLOPE_TREND
   ]
   df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
   # Ensure all numeric
   for col in df.columns:
       df[col] = pd.to_numeric(df[col], errors="coerce")
   return df
# ----------- PREDICTION -----------
if st.button("Predict Churn"):
   try:
       X_df = build_feature_df()
       # Helpful checks (won’t break app)
       if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != X_df.shape[1]:
           st.warning(
               f"Scaler expects {scaler.n_features_in_} features, "
               f"but you provided {X_df.shape[1]}. Check FEATURE_COLUMNS/order."
           )
       if hasattr(model, "n_features_in_") and model.n_features_in_ != X_df.shape[1]:
           st.warning(
               f"Model expects {model.n_features_in_} features, "
               f"but you provided {X_df.shape[1]}. Check FEATURE_COLUMNS/order."
           )
       X_scaled = scaler.transform(X_df)
       y_pred = model.predict(X_scaled)[0]
       label = "Churn" if int(y_pred) == 1 else "Not Churn"
       if hasattr(model, "predict_proba"):
           proba = model.predict_proba(X_scaled)[0][1]
           st.success(f"**Prediction:** {label}  |  **Churn probability:** {proba:.2%}")
       else:
           st.success(f"**Prediction:** {label}")
       with st.expander("Features sent to the model"):
           st.dataframe(X_df)
   except Exception as e:
       st.error(f"Prediction failed: {e}")
