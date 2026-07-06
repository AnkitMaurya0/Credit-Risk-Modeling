import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Feature Importance",
    page_icon="🔍",
    layout="wide"
)

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

RESULT_DIR = BASE_DIR / "results" / "shap"

# =====================================================
# Load CSS
# =====================================================

css_path = BASE_DIR / "assets" / "style.css"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# =====================================================
# Title
# =====================================================

st.markdown(
"""
<div class="main-title">
🔍 Feature Importance & Explainability
</div>

<div class="sub-title">
Understand which features influenced the trained Random Forest model.
</div>
""",
unsafe_allow_html=True
)

st.divider()

# =====================================================
# Feature Importance Chart
# =====================================================

fi = pd.read_csv(DATA_DIR / "feature_importance.csv")

fi = fi.sort_values(
    "Importance",
    ascending=False
).head(10)

fig = px.bar(
    fi,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    title="Top 10 Most Important Features"
)

fig.update_layout(
    height=500,
    yaxis_title="Feature",
    xaxis_title="Importance Score"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# Feature Importance Table
# =====================================================

st.subheader("📋 Top 10 Important Features")

table = fi.copy()

table.index = range(1, len(table) + 1)

st.dataframe(
    table,
    use_container_width=True
)

st.divider()

# =====================================================
# SHAP Summary
# =====================================================

st.subheader("📊 SHAP Summary Plot")

st.image(
    RESULT_DIR / "shap_summary.png",
    use_container_width=True
)

st.info(
"""
SHAP (SHapley Additive exPlanations) explains **why**
the trained Random Forest model made a prediction.

Positive SHAP value
→ increases loan default risk

Negative SHAP value
→ decreases loan default risk
"""
)

st.divider()

# =====================================================
# Global SHAP Importance
# =====================================================

st.subheader("📈 Global SHAP Feature Importance")

st.image(
    RESULT_DIR / "shap_bar.png",
    use_container_width=True
)

st.caption(
"""
The above chart shows the overall importance of every feature
used by the Random Forest model during prediction.
"""
)

st.divider()

# =====================================================
# Feature Categories
# =====================================================

st.subheader("🗂 Features Used by the Model")

col1, col2 = st.columns(2)

with col1:

    st.success("""
### Financial Features

• Annual Income

• Loan Amount

• Interest Rate

• Monthly Installment

• Debt-to-Income Ratio (DTI)

• Revolving Balance

• Credit Utilization
""")

with col2:

    st.success("""
### Credit Profile Features

• Credit Grade

• Verification Status

• Employment Length

• Home Ownership

• Loan Purpose

• Public Records

• Delinquencies

• Credit Inquiries
""")

st.divider()

# =====================================================
# How SHAP Works
# =====================================================

st.subheader("⚙️ How SHAP Explains the Prediction")

st.info(
"""
Step 1 → Applicant details are entered.

Step 2 → Data is preprocessed.

Step 3 → Random Forest predicts loan default probability.

Step 4 → SHAP calculates how much each feature contributed.

Step 5 → Features increasing risk receive positive SHAP values.

Step 6 → Features reducing risk receive negative SHAP values.
"""
)

st.divider()

# =====================================================
# Why SHAP?
# =====================================================

st.success(
"""
### Why SHAP is Used

✔ Makes the Machine Learning model explainable.

✔ Shows why a prediction was made.

✔ Identifies the most influential features.

✔ Improves transparency and trust.

✔ Helps users understand the prediction instead of seeing only a probability.
"""
)