import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Load CSS
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

css_file = BASE_DIR / "assets" / "style.css"

with open(css_file, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# -----------------------------
# HOME PAGE
# -----------------------------

st.markdown(
    """
    <div class="main-title">
    🏦 Credit Risk Prediction Dashboard
    </div>

    <div class="sub-title">
    Machine Learning Based Loan Default Prediction using Random Forest
    </div>
    """,
    unsafe_allow_html=True
)

c1,c2,c3,c4=st.columns(4)

with c1:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    Model
    </div>

    <div class="metric-value">
    Random Forest
    </div>

    </div>
    """,unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    Algorithm
    </div>

    <div class="metric-value">
    Supervised ML
    </div>

    </div>
    """,unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    Explainability
    </div>

    <div class="metric-value">
    SHAP
    </div>

    </div>
    """,unsafe_allow_html=True)

with c4:

    st.markdown("""
    <div class="card">

    <div class="metric-title">
    Dataset
    </div>

    <div class="metric-value">
    LendingClub
    </div>

    </div>
    """,unsafe_allow_html=True)

st.write("")
st.write("")

left,right=st.columns([2,1])

with left:

    st.subheader("📖 Project Overview")

    st.write("""

This project predicts whether a loan applicant is likely to default using Machine Learning.

The system includes:

- Data Cleaning
- SMOTE
- Feature Engineering
- Random Forest
- Hyperparameter Tuning
- SHAP Explainability
- Streamlit Deployment

Use the pages in the left sidebar to:

- Predict Credit Risk
- View Model Performance
- Explore Feature Importance
- Learn about the project

""")

with right:

    st.info("""

### Quick Stats

✔ Random Forest

✔ SMOTE

✔ SHAP

✔ ROC Evaluation

✔ Streamlit Dashboard

""")