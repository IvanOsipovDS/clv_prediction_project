# 🧮 Customer Lifetime Value Prediction

This project predicts the Customer Lifetime Value (CLV) for e-commerce users based on their past purchase behavior. The objective is to help businesses identify high-value customers and make informed marketing decisions.

---

## 📌 Key Highlights

- Predicts **12-month future CLV** using historical RFM features.
- Models compared: **Random Forest**, **XGBoost**, **LightGBM**, **MLP (Neural Network)**.
- Uses **SHAP** for model interpretability at global and individual levels.

---

## 📁 Project Structure

```
clv-prediction/
│
├── data/                        # Raw and sample data (no sensitive info)
├── notebooks/
│   ├── 1_EDA.ipynb              # Feature engineering, target creation
│   ├── 2_modeling.ipynb         # Model training and evaluation
│   └── 3_interpretation.ipynb   # SHAP interpretability
│
├── src/
│   ├── data_preparation/        # Data loading and cleaning
│   ├── feature_engineering/     # RFM and CLV features
│   ├── modeling/                # Model training scripts
│   └── evaluation/              # Metrics and visualization
│
├── models/                      # Trained model files (optional)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/clv-prediction.git
cd clv-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks in Order

- `1_EDA.ipynb`: Feature generation + target labels
- `2_modeling.ipynb`: Train & evaluate regression models
- `3_interpretation.ipynb`: SHAP-based interpretation

---

## 🧠 Models and Performance

| Model        | R² Score |
|--------------|----------|
| Random Forest| ~0.51    |
| XGBoost      | ~0.45    |
| LightGBM     | **~0.56** |
| MLP (Neural) | ~0.52    |

All models were trained to predict **Log-transformed Future CLV**.

---

## 📊 Features Used

- **Recency**: Days since last purchase  
- **Frequency**: Number of purchases  
- **Monetary**: Total spend before cutoff  
- **AveragePurchaseValue**  
- **CustomerLifespan**: First → last purchase  
- **MeanDaysBetweenPurchases**

**Target**:  
- `FutureCLV`: Revenue after cutoff date  
- `LogFutureCLV`: Log-transformed for modeling

---

## 🔍 SHAP Interpretability

- **Global Feature Importance**: Highlights most influential variables across all customers  
- **Local Explanation**: Understand individual prediction drivers

---

## 💡 Future Work

- Feature enhancement (behavioral/temporal data)
- Model improvement via **Stacking/Blending**
- Customer segmentation before modeling
- Time-series CLV estimation

---

## 📎 Author

**Ivan Osipov** 
📍 Based in Buenos Aires
💼 Data Scientist
[LinkedIn](https://www.linkedin.com/in/ivan-osipov-dsml/)
