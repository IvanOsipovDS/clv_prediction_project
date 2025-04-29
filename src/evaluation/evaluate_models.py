# src/evaluation/evaluate_models.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate a regression model using MAE, RMSE, and R2 metrics.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: Dictionary containing MAE, RMSE, and R2 scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    results = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

    return results


def compare_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple regression models based on evaluation metrics.

    Args:
        models (dict): Dictionary of {model_name: trained_model}.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True target values for the test set.

    Returns:
        pd.DataFrame: Evaluation results for all models.
    """
    results = []

    for name, model in models.items():
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            scores = evaluate_regression_model(y_test, y_pred)
            scores['Model'] = name
            results.append(scores)

    return pd.DataFrame(results).set_index('Model')
