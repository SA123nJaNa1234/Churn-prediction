import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load trained pipeline (preprocessing + model)
model = load("models/best_model.pkl")

st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")
st.write(
    "This application predicts whether a customer is likely to churn "
    "based on contract, pricing, and service details."
)

# ---------------- Sidebar Inputs ---------------- #
st.sidebar.header("Customer Information")

def user_input():
    data = {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.sidebar.selectbox(
            "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        ),
        "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),

        "tenure": st.sidebar.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        ),

        "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.sidebar.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        ),
        "InternetService": st.sidebar.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        ),
        "OnlineSecurity": st.sidebar.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        ),
        "OnlineBackup": st.sidebar.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        ),
        "DeviceProtection": st.sidebar.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        ),
        "TechSupport": st.sidebar.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        ),
        "StreamingTV": st.sidebar.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        ),
        "StreamingMovies": st.sidebar.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        ),

        "Contract": st.sidebar.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        ),
        "PaperlessBilling": st.sidebar.selectbox(
            "Paperless Billing", ["Yes", "No"]
        ),
        "PaymentMethod": st.sidebar.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        ),

        "MonthlyCharges": st.sidebar.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        ),
        "TotalCharges": st.sidebar.number_input(
            "Total Charges", min_value=0.0, value=1000.0
        )
    }

    return pd.DataFrame([data])


input_df = user_input()

# ---------------- Prediction ---------------- #
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Customer is likely to churn\n\n"
            f"**Churn Probability:** {probability:.2f}"
        )
    else:
        st.success(
            f"✅ Customer is unlikely to churn\n\n"
            f"**Retention Probability:** {1 - probability:.2f}"
        )

# ---------------- Explainability ---------------- #
st.subheader("Model Explainability (Global)")

classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocess"]
feature_names = preprocessor.get_feature_names_out()

if hasattr(classifier, "coef_"):
    # Logistic Regression
    coefficients = classifier.coef_[0]
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": coefficients
    }).sort_values(by="Impact", ascending=False)

    st.write("Features increasing churn risk:")
    st.dataframe(coef_df.head(10))

    st.write("Features reducing churn risk:")
    st.dataframe(coef_df.tail(10))

elif hasattr(classifier, "feature_importances_"):
    # Random Forest
    importances = classifier.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.write("Top churn drivers:")
    st.dataframe(imp_df.head(10))
