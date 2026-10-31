"""Streamlit app: predict telecom customer churn.

Uses the three artefacts saved by the notebook:
    churn_model.pkl  - tuned Logistic Regression
    scaler.pkl       - StandardScaler fitted on the training data
    threshold.pkl    - decision threshold (0.40)

Place them in a `model/` folder next to this file (or next to it directly).
Run with:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

BASE_DIR = Path(__file__).parent
SEARCH_DIRS = [BASE_DIR / "model", BASE_DIR]
ARTEFACTS = ("churn_model.pkl", "scaler.pkl", "threshold.pkl")

# Column order the model was trained on (matches the notebook's X_train).
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "MonthlyCharges", "InternetService_Fiber optic", "InternetService_No",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


def find_file(name):
    for folder in SEARCH_DIRS:
        path = folder / name
        if path.exists():
            return path
    return None


@st.cache_resource
def load_artefacts():
    paths = {name: find_file(name) for name in ARTEFACTS}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        return None, None, None, missing
    model = joblib.load(paths["churn_model.pkl"])
    scaler = joblib.load(paths["scaler.pkl"])
    threshold = float(joblib.load(paths["threshold.pkl"]))
    return model, scaler, threshold, []


def yes_no(value):
    return 1 if value == "Yes" else 0


def build_features(inp, columns):
    """Turn the form inputs into one row with the same encoding as the notebook."""
    has_phone = inp["phone"] == "Yes"
    has_internet = inp["internet"] != "No"

    def addon(value):
        # 'No internet service' was merged into 'No' (0) in the notebook.
        return yes_no(value) if has_internet else 0

    row = {
        "gender": 1 if inp["gender"] == "Male" else 0,
        "SeniorCitizen": yes_no(inp["senior"]),
        "Partner": yes_no(inp["partner"]),
        "Dependents": yes_no(inp["dependents"]),
        "tenure": inp["tenure"],
        "PhoneService": yes_no(inp["phone"]),
        # 'No phone service' was merged into 'No' (0) in the notebook.
        "MultipleLines": yes_no(inp["multiple_lines"]) if has_phone else 0,
        "OnlineSecurity": addon(inp["security"]),
        "OnlineBackup": addon(inp["backup"]),
        "DeviceProtection": addon(inp["protection"]),
        "TechSupport": addon(inp["tech_support"]),
        "StreamingTV": addon(inp["streaming_tv"]),
        "StreamingMovies": addon(inp["streaming_movies"]),
        "PaperlessBilling": yes_no(inp["paperless"]),
        "MonthlyCharges": inp["monthly"],
        "InternetService_Fiber optic": int(inp["internet"] == "Fiber optic"),
        "InternetService_No": int(inp["internet"] == "No"),
        "Contract_One year": int(inp["contract"] == "One year"),
        "Contract_Two year": int(inp["contract"] == "Two year"),
        "PaymentMethod_Credit card (automatic)": int(inp["payment"] == "Credit card (automatic)"),
        "PaymentMethod_Electronic check": int(inp["payment"] == "Electronic check"),
        "PaymentMethod_Mailed check": int(inp["payment"] == "Mailed check"),
    }
    return pd.DataFrame([row])[columns]


# ---------------------------------------------------------------- page
st.title("📉 Customer Churn Predictor")
st.caption(
    "Estimates how likely a telecom customer is to leave, using a Logistic "
    "Regression model trained on the IBM Telco Customer Churn dataset."
)

model, scaler, threshold, missing = load_artefacts()
if missing:
    st.error(
        "Model files not found: " + ", ".join(f"`{m}`" for m in missing) + ".\n\n"
        "Run the notebook's *Save the model* cells, then put the three `.pkl` "
        "files in a `model/` folder next to `app.py`."
    )
    st.stop()

columns = list(getattr(scaler, "feature_names_in_", FEATURE_ORDER))

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Customer details")

    st.markdown("**Account**")
    c1, c2 = st.columns(2)
    tenure = c1.slider("Tenure (months)", 0, 72, 12)
    monthly = c2.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = c2.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless = c1.selectbox("Paperless billing", ["Yes", "No"])

    st.markdown("**Demographics**")
    d1, d2, d3, d4 = st.columns(4)
    gender = d1.selectbox("Gender", ["Female", "Male"])
    senior = d2.selectbox("Senior citizen", ["No", "Yes"])
    partner = d3.selectbox("Partner", ["No", "Yes"])
    dependents = d4.selectbox("Dependents", ["No", "Yes"])

    st.markdown("**Services**")
    s1, s2 = st.columns(2)
    phone = s1.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = s2.selectbox("Multiple lines", ["No", "Yes"], disabled=(phone == "No"))
    internet = s1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    no_net = internet == "No"
    a1, a2, a3 = st.columns(3)
    security = a1.selectbox("Online security", ["No", "Yes"], disabled=no_net)
    backup = a2.selectbox("Online backup", ["No", "Yes"], disabled=no_net)
    protection = a3.selectbox("Device protection", ["No", "Yes"], disabled=no_net)
    tech_support = a1.selectbox("Tech support", ["No", "Yes"], disabled=no_net)
    streaming_tv = a2.selectbox("Streaming TV", ["No", "Yes"], disabled=no_net)
    streaming_movies = a3.selectbox("Streaming movies", ["No", "Yes"], disabled=no_net)

inputs = {
    "gender": gender, "senior": senior, "partner": partner, "dependents": dependents,
    "tenure": tenure, "phone": phone, "multiple_lines": multiple_lines,
    "internet": internet, "security": security, "backup": backup,
    "protection": protection, "tech_support": tech_support,
    "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
    "contract": contract, "paperless": paperless, "payment": payment,
    "monthly": monthly,
}

features = build_features(inputs, columns)
scaled = pd.DataFrame(scaler.transform(features), columns=columns)
probability = float(model.predict_proba(scaled)[0, 1])
high_risk = probability >= threshold

with right:
    st.subheader("Prediction")
    st.metric("Churn probability", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    if high_risk:
        st.error(f"**High churn risk** (at or above the {threshold:.2f} threshold)")
    else:
        st.success(f"**Low churn risk** (below the {threshold:.2f} threshold)")
    st.caption(
        f"The model flags a customer as likely to churn when the probability is "
        f"{threshold:.2f} or higher. This threshold favours catching churners over "
        f"avoiding false alarms."
    )

    st.markdown("**What is pushing this prediction**")
    contributions = pd.Series(model.coef_[0] * scaled.iloc[0].values, index=columns)
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(6)
    st.bar_chart(top.sort_values())
    st.caption(
        "Contribution of each feature to the log-odds of churn. Positive values "
        "raise the risk; negative values lower it."
    )

st.divider()
st.caption(
    "For learning and demonstration only. Predictions come from a single "
    "historical dataset and show association, not causation."
)
