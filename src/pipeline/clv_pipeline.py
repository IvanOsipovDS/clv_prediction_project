# src/pipeline/clv_pipeline.py

import pandas as pd

# Import modules from src
from data_preparation.prepare_data import load_raw_data, clean_data
from feature_engineering.engineer_features import create_rfm_features
from modeling.train_models import split_data, train_random_forest, train_xgboost, train_lightgbm, train_mlp
from evaluation.evaluate_models import compare_models
from explainability.shap_analysis import compute_shap_values, plot_shap_summary

def run_clv_pipeline(data_path: str, customer_id_col: str, invoice_date_col: str, amount_col: str):
    """
    Full pipeline to train models for CLV prediction and evaluate them.

    Args:
        data_path (str): Path to the raw transaction data CSV file.
        customer_id_col (str): Name of the customer ID column.
        invoice_date_col (str): Name of the transaction date column.
        amount_col (str): Name of the transaction amount column.

    Returns:
        pd.DataFrame: Comparison table of model evaluation results.
    """

    # Step 1: Load and clean data
    print("Loading and cleaning data...")
    raw_data = load_raw_data(data_path)
    clean_df = clean_data(raw_data)

    # Step 2: Feature Engineering
    print("Generating RFM features...")
    features_df = create_rfm_features(clean_df, customer_id_col, invoice_date_col, amount_col)

    # Step 3: Prepare X and y
    X = features_df.drop(columns=[amount_col], errors='ignore')  # remove if exists
    y = features_df['Monetary']  # Using Monetary as a proxy for CLV for now

    # Step 4: Train-test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Train models
    print("Training models...")
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    lgbm_model = train_lightgbm(X_train, y_train)
    mlp_model = train_mlp(X_train, y_train)

    models = {
        'RandomForest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model,
        'MLP': mlp_model
    }

    # Step 6: Evaluate models
    print("Evaluating models...")
    evaluation_results = compare_models(models, X_test, y_test)

    print("\nModel evaluation results:")
    print(evaluation_results)

    # Step 7: Explain best model
    best_model_name = evaluation_results['RMSE'].idxmin()
    best_model = models[best_model_name]

    print(f"\nExplaining the best model: {best_model_name}...")
    explainer, shap_values = compute_shap_values(best_model, X_train)
    plot_shap_summary(shap_values, X_train, plot_type="bar")

    return evaluation_results

# Example of usage (you can remove or comment this before production):
# if __name__ == "__main__":
#     run_clv_pipeline("data/raw/your_data.csv", "CustomerID", "InvoiceDate", "Amount")
