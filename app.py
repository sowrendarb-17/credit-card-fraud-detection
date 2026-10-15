import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Load saved model and feature names
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "xgboost_fraud_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


st.title("💳 Credit Card Fraud Detection")
st.write(
    "This application uses a trained XGBoost model "
    "to classify credit card transactions."
)

try:
    model, model_features = load_model()
except Exception as error:
    st.error(
        "Could not load the model files. Make sure "
        "xgboost_fraud_model.pkl and model_features.pkl "
        "are in the same folder as app.py."
    )
    st.stop()

st.success("XGBoost model loaded successfully!")

st.info(
    "Upload transaction data containing the same 30 features "
    "used during model training. This app does not retrain the model."
)

# --------------------------------------------------
# Upload transaction CSV
# --------------------------------------------------
st.header("📂 Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Transaction Data Preview")
        st.dataframe(data.head(10), use_container_width=True)

        # Remove target column if included in the uploaded data
        input_data = data.drop(columns=["Class"], errors="ignore")

        # Check that all expected features are present
        missing_features = [
            feature for feature in model_features
            if feature not in input_data.columns
        ]

        if missing_features:
            st.error(
                "Your CSV is missing these required features: "
                + ", ".join(missing_features)
            )
        else:
            # Keep features in the exact training order
            input_data = input_data[model_features]

            if input_data.empty:
                st.warning("The uploaded CSV contains no transactions.")
            elif not all(
                pd.api.types.is_numeric_dtype(input_data[col])
                for col in input_data.columns
            ):
                st.error("All model input features must be numeric.")
            else:
                if input_data.isnull().any().any():
                    st.error(
                        "Your CSV contains missing values. "
                        "Please handle them using the same preprocessing "
                        "method used during model training."
                    )
                else:
                    if not input_data.map(
                        lambda value: pd.notna(value)
                    ).all().all():
                        st.error("The CSV contains invalid values.")
                    else:
                        predictions = model.predict(input_data)
                        probabilities = model.predict_proba(input_data)[:, 1]

                        results = data.copy()
                        results["Predicted Class"] = predictions
                        results["Fraud Probability"] = probabilities
                        results["Prediction"] = [
                            "🚨 Potential Fraud" if prediction == 1
                            else "✅ Likely Legitimate"
                            for prediction in predictions
                        ]

                        st.subheader("🔍 Prediction Results")
                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        fraud_count = int((predictions == 1).sum())
                        legitimate_count = int((predictions == 0).sum())

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Potential Fraud Transactions",
                            fraud_count
                        )

                        col2.metric(
                            "Likely Legitimate Transactions",
                            legitimate_count
                        )

                        csv_output = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="⬇️ Download Prediction Results",
                            data=csv_output,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )

                        st.caption(
                            "Predictions are model estimates, not proof "
                            "that a transaction is fraudulent or safe."
                        )

    except Exception as error:
        st.error(f"Error processing the CSV file: {error}")

# --------------------------------------------------
# Manual transaction input
# --------------------------------------------------
st.divider()
st.header("🧪 Test a Single Transaction")

st.write(
    "Enter values for every feature. Use the same feature definitions, "
    "units, and preprocessing as the training dataset."
)

with st.expander("Enter transaction feature values"):
    with st.form("transaction_form"):
        manual_values = {}

        columns = st.columns(3)

        for index, feature in enumerate(model_features):
            with columns[index % 3]:
                manual_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.8f",
                    key=f"feature_{index}"
                )

        submitted = st.form_submit_button(
            "Predict Transaction"
        )

    if submitted:
        manual_data = pd.DataFrame(
            [[manual_values[feature] for feature in model_features]],
            columns=model_features
        )

        prediction = int(model.predict(manual_data)[0])
        fraud_probability = float(
            model.predict_proba(manual_data)[0][1]
        )

        st.metric(
            "Estimated Fraud Probability",
            f"{fraud_probability:.6%}"
        )

        if prediction == 1:
            st.error("🚨 Model prediction: Potential Fraud")
        else:
            st.success("✅ Model prediction: Likely Legitimate")

        st.caption(
            "All-zero inputs are only a software test, not a realistic "
            "transaction. Use actual feature values for meaningful results."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Educational machine learning project | "
    "XGBoost | Credit Card Fraud Detection"
)
