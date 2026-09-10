import streamlit as st
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/linear_regression_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Training feature order (must match Day 20 exactly)
FEATURE_COLS = [
    "age", "bmi", "children",
    "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest",
]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("Enter customer information below to estimate the medical insurance cost.")
st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
    bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=22.5, step=0.1, format="%.1f")
    children = st.number_input("Number of Children", min_value=0, max_value=20, value=0, step=1)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoking Status", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Insurance Cost", use_container_width=True, type="primary"):

    # Validation
    errors = []
    if age <= 0:
        errors.append("Age must be greater than 0.")
    if bmi <= 0:
        errors.append("BMI must be greater than 0.")
    if children < 0:
        errors.append("Number of children cannot be negative.")

    if errors:
        for err in errors:
            st.error(err)
    else:
        try:
            # Build input row with same encoding as training (drop_first=True)
            row = {
                "age":              age,
                "bmi":              bmi,
                "children":         children,
                "sex_male":         1 if gender == "Male"      else 0,
                "smoker_yes":       1 if smoker == "Yes"       else 0,
                "region_northwest": 1 if region == "Northwest" else 0,
                "region_southeast": 1 if region == "Southeast" else 0,
                "region_southwest": 1 if region == "Southwest" else 0,
            }

            # Align columns to training order
            input_df = pd.DataFrame([row])[FEATURE_COLS]

            # Predict
            prediction = model.predict(input_df)[0]

            # Show result
            st.success("Prediction complete!")
            st.metric(
                label="Estimated Insurance Cost",
                value=f"${prediction:,.2f}",
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Medical Insurance Cost Prediction · Linear Regression Model · Day 22")
