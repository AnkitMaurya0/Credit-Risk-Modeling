import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(

    page_title="Model Performance",

    page_icon="📈",

    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"

css_path = BASE_DIR / "assets" / "style.css"

if css_path.exists():

    with open(css_path,"r",encoding="utf-8") as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True
        )

st.markdown(
"""
<div class="main-title">

📈 Model Performance Dashboard

</div>

<div class="sub-title">

Performance evaluation of the trained Random Forest model.

</div>
""",
unsafe_allow_html=True
)


results = pd.read_csv(

    DATA_DIR / "model_results.csv"

)

# ============================================================
# Best Model Metrics
# ============================================================

rf = results[results["Model"] == "Random Forest"].iloc[0]

st.divider()

st.subheader("🏆 Best Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Accuracy",
        f"{rf['Accuracy']:.2%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{rf['Precision']:.2%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{rf['Recall']:.2%}"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{rf['F1_Score']:.2%}"
    )

with col5:
    st.metric(
        "ROC-AUC",
        f"{rf['ROC_AUC']:.3f}"
    )


# ============================================================
# Model Comparison
# ============================================================

st.divider()

st.subheader("📊 Model Comparison")

metric = st.selectbox(

    "Select Performance Metric",

    [

        "Accuracy",

        "Precision",

        "Recall",

        "F1_Score",

        "ROC_AUC"

    ]

)

fig = px.bar(

    results,

    x="Model",

    y=metric,

    color="Model",

    text=metric,

    title=f"Comparison of {metric} Across Models"

)

fig.update_traces(

    texttemplate="%{text:.3f}",

    textposition="outside"

)

fig.update_layout(

    height=500,

    showlegend=False,

    yaxis_title=metric,

    xaxis_title="Machine Learning Models"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

st.divider()

st.subheader("⚙️ Model Training Pipeline")

st.info("""
Raw LendingClub Dataset
        ↓
Data Cleaning
        ↓
Missing Value Imputation
        ↓
One Hot Encoding
        ↓
SMOTE (Training Only)
        ↓
Random Forest Training
        ↓
Model Evaluation
""")

best = results.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]

st.success(f"""
### 🏆 Selected Model

**{best['Model']}** was selected for deployment.

Reasons:

• Highest Accuracy

• Good ROC-AUC

• Stable Performance

• Works well after SMOTE balancing
""")
fi = pd.read_csv(
    DATA_DIR/"feature_importance.csv"
)

fi = fi.head(10)

fig = px.bar(

    fi,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    title="Top 10 Important Features"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("🔍 SHAP Explainability")

st.image(
    BASE_DIR/"results/shap/shap_summary.png",
    use_container_width=True
)

st.image(
    BASE_DIR/"results/shap/shap_bar.png",
    use_container_width=True
)