# 🧮 Customer Lifetime Value Prediction

This project aims to predict the Customer Lifetime Value (CLV) based on customer transactions and survey data. The goal is to help businesses identify high-value customers and allocate marketing resources efficiently.

## 📌 Key Highlights

- Predicts **Future CLV** for a 12-month window using historical data.
- Combines **RFM** features and engineered time-based indicators.
- Compares multiple regression models: **Random Forest**, **XGBoost**, **LightGBM**, and **Neural Network (MLP)**.
- Applies **SHAP** for global and local interpretability.

---

## 📁 Project Structure

clv-prediction/
│
├── data/ # Raw and sample data files (excluding sensitive)
├── notebooks/
│ ├── 1_EDA.ipynb # Data cleaning, RFM calculation, target generation
│ ├── 2_modeling.ipynb # Model training and evaluation
│ └── 3_interpretation.ipynb # SHAP analysis and interpretability
│
├── src/
│ ├── data_preparation/ # Loading and cleaning scripts
│ ├── feature_engineering/ # RFM and CLV target computation
│ ├── modeling/ # Training and prediction logic
│ └── evaluation/ # Metrics and comparison utilities
│
├── models/ # Trained model files (optional)
├── requirements.txt # Dependencies
└── README.md

---

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/yourusername/clv-prediction.git
cd clv-prediction
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run Notebooks in Order
1_EDA.ipynb → Data prep, feature engineering, and target creation

2_modeling.ipynb → Train/test split, modeling, evaluation

3_interpretation.ipynb → SHAP-based model interpretability

🧠 Models Used
Model       	  R² Score
Random Forest	   ~0.532
XGBoost	         ~0.514
LightGBM	       ~0.534
MLP (Neural)	   ~0.560

All models were trained on historical RFM-style features to predict future 12-month CLV.

📊 Feature Engineering
The dataset includes:

RFM Metrics: Recency, Frequency, Monetary

Derived Metrics:

Average Purchase Value

Customer Lifespan

Mean Days Between Purchases

Targets were calculated as:

FutureCLV: Future revenue after cutoff

LogFutureCLV: Log-transformed target used for modeling

📈 Interpretation
Using SHAP (SHapley Additive exPlanations):

Global importance: Frequency, Recency, and Lifespan were most impactful

Local explanations: Individual predictions can be traced to feature influence

<!-- Optional image preview -->

💡 Future Improvements
Add new features (e.g. session behavior, geo location)

Try stacking/ensemble techniques

Segment customers and customize predictions per group

📎 Author
Ivan Osipov

LinkedIn

Portfolio

Email
