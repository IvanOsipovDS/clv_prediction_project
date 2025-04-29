# src/explainability/shap_analysis.py

import shap
import pandas as pd
import matplotlib.pyplot as plt

def compute_shap_values(model, X_train: pd.DataFrame):
    """
    Compute SHAP values for a fitted model and training data.

    Args:
        model: Trained model that supports SHAP.
        X_train (pd.DataFrame): Training feature set.

    Returns:
        explainer: SHAP explainer object.
        shap_values: SHAP values array.
    """
    # Select appropriate SHAP explainer based on model type
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    return explainer, shap_values


def plot_shap_summary(shap_values, X_train: pd.DataFrame, plot_type: str = "bar"):
    """
    Plot a SHAP summary plot to visualize feature importance.

    Args:
        shap_values: Computed SHAP values.
        X_train (pd.DataFrame): Training feature set.
        plot_type (str): Type of SHAP plot ('bar' or 'dot').

    Returns:
        None
    """
    plt.title(f"SHAP Summary Plot ({plot_type} plot)")
    shap.summary_plot(shap_values, features=X_train, plot_type=plot_type)
