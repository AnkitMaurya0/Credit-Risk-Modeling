# ============================================================
# AI-Based Credit Risk Prediction System
# Prediction Page
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
ASSET_DIR = BASE_DIR / "assets"

MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# ============================================================
# Helper Functions
# ============================================================


def load_css():
    css_file = ASSET_DIR / "style.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def calculate_emi(principal, annual_rate, months):
    """Calculate the monthly EMI for the loan."""
    if annual_rate == 0:
        return principal / months

    monthly_rate = annual_rate / (12 * 100)
    emi = (
        principal * monthly_rate * (1 + monthly_rate) ** months
    ) / ((1 + monthly_rate) ** months - 1)
    return emi


def credit_score_to_grade(score):
    """Convert credit score into a simplified grade and sub-grade."""
    if score >= 750:
        return "A", "A2"
    if score >= 700:
        return "B", "B2"
    if score >= 650:
        return "C", "C3"
    if score >= 600:
        return "D", "D3"
    if score >= 550:
        return "E", "E3"
    if score >= 500:
        return "F", "F4"
    return "G", "G5"


def calculate_approval_probability(default_probability):
    return 100 - default_probability


def get_risk_level(default_probability):
    if default_probability < 30:
        return "🟢 Low Risk"
    if default_probability < 60:
        return "🟡 Moderate Risk"
    return "🔴 High Risk"


def build_input_frame(values):
    return pd.DataFrame(
        {
            "loan_amnt": [values["loan_amnt"]],
            "term": [values["term"]],
            "int_rate": [values["int_rate"]],
            "installment": [values["installment"]],
            "grade": [values["grade"]],
            "sub_grade": [values["sub_grade"]],
            "emp_length": [values["emp_length"]],
            "home_ownership": [values["home_ownership"]],
            "annual_inc": [values["annual_inc"]],
            "verification_status": [values["verification_status"]],
            "purpose": [values["purpose"]],
            "dti": [values["dti"]],
            "delinq_2yrs": [values["delinq_2yrs"]],
            "inq_last_6mths": [values["inq_last_6mths"]],
            "open_acc": [values["open_acc"]],
            "pub_rec": [0],
            "revol_bal": [values["loan_amnt"] * values["revol_util"] / 100],
            "revol_util": [values["revol_util"]],
            "total_acc": [values["total_acc"]],
            "mort_acc": [1 if values["home_ownership"] == "MORTGAGE" else 0],
            "pub_rec_bankruptcies": [0],
            "application_type": ["Individual"],
        }
    )


def render_header():
    st.markdown(
        """
        <div class="main-title">
        🏦 AI-Based Credit Risk Prediction System
        </div>

        <div class="sub-title">
        Predict the probability of loan default using a trained Machine Learning model.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_form():
    st.header("📝 Applicant Information")

    left, right = st.columns(2)

    with left:
        loan_amnt = st.number_input(
            "Loan Amount (₹)",
            min_value=50000,
            max_value=10000000,
            value=500000,
            step=10000,
        )
        annual_inc = st.number_input(
            "Annual Income (₹)",
            min_value=100000,
            max_value=50000000,
            value=800000,
            step=50000,
        )
        credit_score = st.slider("Credit Score", 300, 900, 700)
        int_rate = st.slider("Interest Rate (%)", 5.0, 35.0, 12.5, 0.1)
        term = st.selectbox("Loan Tenure", ["36 months", "60 months"])
        emp_length = st.selectbox(
            "Employment Experience",
            [
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years",
            ],
        )
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    with right:
        verification_status = st.selectbox(
            "Income Verification",
            ["Verified", "Source Verified", "Not Verified"],
        )
        purpose = st.selectbox(
            "Loan Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "medical",
                "car",
                "vacation",
                "house",
                "moving",
                "other",
            ],
        )
        dti = st.slider("Debt-To-Income Ratio (%)", 0.0, 40.0, 18.0)
        delinq_2yrs = st.number_input("Missed Payments (Last 2 Years)", 0, 20, 0)
        inq_last_6mths = st.number_input("Loan Inquiries (Last 6 Months)", 0, 20, 1)
        open_acc = st.number_input("Open Credit Accounts", 1, 50, 5)
        total_acc = st.number_input("Total Credit Accounts", 1, 100, 12)
        revol_util = st.slider("Credit Utilization (%)", 0.0, 100.0, 35.0)

    return {
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "credit_score": credit_score,
        "int_rate": int_rate,
        "term": term,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "total_acc": total_acc,
        "revol_util": revol_util,
    }


def render_prediction_results(default_probability, approval_probability, confidence, risk_level):
    st.divider()
    st.header("📊 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Level", risk_level)
    with col2:
        st.metric("Default Risk", f"{default_probability:.2f}%")
    with col3:
        st.metric("Loan Approval", f"{approval_probability:.2f}%")
    with col4:
        st.metric("Model Confidence", f"{confidence:.2f}%")

    st.write("")

    gauge1, gauge2 = st.columns(2)
    with gauge1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=default_probability,
                number={"suffix": "%"},
                title={"text": "Chance of Loan Default"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 30], "color": "green"},
                        {"range": [30, 60], "color": "gold"},
                        {"range": [60, 100], "color": "red"},
                    ],
                },
            )
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with gauge2:
        fig2 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=approval_probability,
                number={"suffix": "%"},
                title={"text": "Chance of Loan Approval"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 40], "color": "red"},
                        {"range": [40, 70], "color": "gold"},
                        {"range": [70, 100], "color": "green"},
                    ],
                },
            )
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    if default_probability < 30:
        st.success(
            "✅ Low Credit Risk\n\nThe trained Random Forest model predicts a LOW probability of loan default."
        )
    elif default_probability < 60:
        st.warning(
            "⚠ Moderate Credit Risk\n\nThe model predicts a MODERATE probability of loan default."
        )
    else:
        st.error(
            "❌ High Credit Risk\n\nThe model predicts a HIGH probability of loan default."
        )

    st.divider()
    st.subheader("🧠 Model Explanation")
    st.info(
        """
        The prediction shown above is generated using the trained Random Forest model.

        SMOTE was used during model training to balance the dataset.

        SHAP explainability was performed during model evaluation (Notebook 6).

        The saved SHAP charts will be displayed in the Feature Importance page.
        """
    )


def render_loan_decision(default_probability, approval_probability):
    st.divider()
    st.subheader("🏦 Loan Decision")

    if default_probability < 30:
        decision = "✅ Likely Approved"
        verification = "Basic Document Verification"
        decision_color = "success"
    elif default_probability < 60:
        decision = "🟡 Needs Additional Verification"
        verification = "Income & Credit Verification"
        decision_color = "warning"
    else:
        decision = "🔴 High Default Risk"
        verification = "Detailed Financial Review"
        decision_color = "error"

    if decision_color == "success":
        st.success(
            f"""
            ### {decision}

            **Estimated Loan Approval Chance:** {approval_probability:.2f}%

            **Next Step:** {verification}

            The applicant appears to have a relatively low predicted default risk.
            """
        )
    elif decision_color == "warning":
        st.warning(
            f"""
            ### {decision}

            **Estimated Loan Approval Chance:** {approval_probability:.2f}%

            **Next Step:** {verification}

            The application may require additional checks before a lending decision.
            """
        )
    else:
        st.error(
            f"""
            ### {decision}

            **Estimated Loan Approval Chance:** {approval_probability:.2f}%

            **Next Step:** {verification}

            The model predicts a higher probability of loan default.
            """
        )


def render_model_info():
    st.divider()
    st.subheader("ℹ️ Model Information")

    left, right = st.columns(2)
    with left:
        st.info(
            """
            **Machine Learning Model**

            • Random Forest Classifier

            • Trained on LendingClub Dataset

            • Hyperparameter Tuned

            • Probability from predict_proba()
            """
        )

    with right:
        st.info(
            """
            **Training Pipeline**

            Dataset

            ↓

            Preprocessing

            ↓

            SMOTE

            ↓

            Random Forest

            ↓

            Prediction

            SMOTE was used only during training.
            """
        )

    st.divider()
    st.caption(
        """
        This prediction is generated using the trained Random Forest model.

        It estimates the probability of loan default.

        Final loan approval depends on the lender's policies, document verification, and other financial checks.
        """
    )


def main():
    load_css()
    model, preprocessor = load_model()
    render_header()

    form_values = render_form()

    months = 36 if form_values["term"] == "36 months" else 60
    installment = calculate_emi(form_values["loan_amnt"], form_values["int_rate"], months)
    grade, sub_grade = credit_score_to_grade(form_values["credit_score"])

    st.info(
        f"""
        **Estimated EMI : ₹{installment:,.0f}**

        Credit Grade Assigned : **{grade} ({sub_grade})**
        """
    )

    predict = st.button("🚀 Check My Loan Risk", use_container_width=True)

    if predict:
        input_df = build_input_frame(
            {
                **form_values,
                "installment": installment,
                "grade": grade,
                "sub_grade": sub_grade,
                "revol_bal": form_values["loan_amnt"] * form_values["revol_util"] / 100,
            }
        )

        X = preprocessor.transform(input_df)
        probability = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]

        default_probability = probability[1] * 100
        approval_probability = calculate_approval_probability(default_probability)
        confidence = probability.max() * 100
        risk_level = get_risk_level(default_probability)

        st.session_state["approval_probability"] = approval_probability
        st.session_state["prediction"] = prediction
        st.session_state["confidence"] = confidence
        st.session_state["risk_level"] = risk_level
        st.session_state["input_df"] = input_df

        render_prediction_results(
            default_probability,
            approval_probability,
            confidence,
            risk_level,
        )
        render_loan_decision(default_probability, approval_probability)

    render_model_info()


if __name__ == "__main__":
    main()
