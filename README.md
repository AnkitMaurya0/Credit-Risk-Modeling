# 🏦 AI-Based Credit Risk Prediction System

<p align="center">

<img src="https://img.shields.io/badge/Python-3.12-blue?logo=python">
<img src="https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit">
<img src="https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange?logo=scikitlearn">
<img src="https://img.shields.io/badge/Explainable-AI-SHAP-success">
<img src="https://img.shields.io/badge/Status-Live-brightgreen">

</p>

An interactive **Machine Learning web application** that predicts the probability of **loan default** using a trained **Random Forest Classifier**.

The project combines **Machine Learning**, **Explainable AI (SHAP)**, and a modern **Streamlit dashboard** to help understand credit risk predictions.

---

# 🌐 Live Demo

### 🚀 Try the application here

https://credit-risk-modeling-zjcg9jpcnu2dcddb8ims3c.streamlit.app/

---

# 📸 Project Screenshots

## 🏠 Home Dashboard

<img src="screenshots/home.png">

---

## 📝 Loan Prediction

<img src="screenshots/prediction.png">

---

## 📊 Model Performance Dashboard

<img src="screenshots/performance.png">

---

## 🔍 Feature Importance & SHAP

<img src="screenshots/feature_importance.png">

---

## ℹ️ About Project

<img src="screenshots/about.png">

---

# 🎯 Project Objective

The objective of this project is to build an intelligent system that predicts whether a loan applicant is likely to default based on financial and credit-related information.

Instead of providing only a prediction, the application also explains **why** the model made that decision using **SHAP Explainability**.

---

# ✨ Features

- 🏦 Loan Default Prediction
- 📈 Loan Approval Probability
- 🎯 Random Forest Classifier
- 🔍 SHAP Explainability
- 📊 Interactive Plotly Charts
- 📉 Model Performance Dashboard
- 📋 Feature Importance Analysis
- 🌙 Professional Dark Theme UI
- ⚡ Fast Streamlit Deployment

---

# 🤖 Machine Learning Workflow

```text
Loan Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Data Preprocessing
      │
      ▼
Missing Value Imputation
      │
      ▼
One-Hot Encoding
      │
      ▼
SMOTE
      │
      ▼
Random Forest Classifier
      │
      ▼
Prediction
      │
      ▼
SHAP Explainability
```

---

# 📂 Project Structure

```text
Credit-Risk-Modeling/

│
├── app.py
├── requirements.txt
├── README.md
│
├── assets/
│     └── style.css
│
├── pages/
│     ├── 1_Prediction.py
│     ├── 2_Model_Performance.py
│     ├── 3_Feature_Importance.py
│     └── 4_About.py
│
├── models/
│     ├── best_model.pkl
│     └── preprocessor.pkl
│
├── notebook/
│     ├── 01_Data_Understanding.ipynb
│     ├── 02_EDA.ipynb
│     ├── 03_Preprocessing.ipynb
│     ├── 04_Model_Training.ipynb
│     ├── 05_Model_Evaluation.ipynb
│     └── 06_SHAP_Analysis.ipynb
│
├── data/
│     └── processed/
│
├── results/
│     └── shap/
│
└── screenshots/
      ├── home.png
      ├── prediction.png
      ├── performance.png
      ├── feature_importance.png
      └── about.png
```

---

# 📊 Dataset

**Dataset Used:** LendingClub Loan Dataset

The dataset contains financial and credit information such as:

- Annual Income
- Loan Amount
- Interest Rate
- Debt-to-Income Ratio (DTI)
- Employment Length
- Home Ownership
- Verification Status
- Credit Grade
- Loan Purpose
- Public Records
- Delinquencies
- Revolving Balance

---

# 🧠 Machine Learning Model

### Algorithm

- Random Forest Classifier

### Data Preprocessing

- Missing Value Imputation
- One-Hot Encoding
- Feature Engineering

### Imbalanced Data Handling

- SMOTE (Synthetic Minority Oversampling Technique)

### Explainability

- SHAP (SHapley Additive exPlanations)

---

# 📈 Model Evaluation

The model was evaluated using multiple metrics.

| Metric | Purpose |
|---------|----------|
| Accuracy | Overall Correct Predictions |
| Precision | Positive Prediction Quality |
| Recall | Ability to Detect Defaults |
| F1 Score | Precision & Recall Balance |
| ROC-AUC | Model Discrimination Ability |

---

# 🔍 Explainable AI (SHAP)

The dashboard integrates SHAP to explain model predictions.

SHAP helps identify:

- Features increasing loan default risk
- Features reducing loan default risk
- Overall feature importance
- Local prediction explanation
- Global model behaviour

---

# 💻 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Streamlit | Web Dashboard |
| Pandas | Data Analysis |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| Imbalanced-Learn | SMOTE |
| SHAP | Explainable AI |
| Plotly | Interactive Charts |
| Matplotlib | Data Visualization |
| Joblib | Model Serialization |

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/AnkitMaurya0/Credit-Risk-Modeling.git
```

Move into the project directory

```bash
cd Credit-Risk-Modeling
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

- XGBoost Implementation
- LightGBM Model
- User Authentication
- Database Integration
- PDF Report Generation
- Real Credit Bureau API
- Loan Recommendation System
- Cloud Database Support

---

# ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The prediction generated by this application is intended to assist decision-making and should **not** be considered as the final loan approval decision.

Actual approval depends on lender policies, financial verification, regulatory requirements, and additional risk assessment.

---

# 👨‍💻 Author

## Ankit Maurya

**B.Tech – Artificial Intelligence & Machine Learning**

GitHub

https://github.com/AnkitMaurya0

---

## ⭐ If you like this project

Please consider giving this repository a **Star ⭐**.

It motivates me to build more Machine Learning projects.
