# src/modeling/train_models.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        test_size (float): Fraction of the dataset to include in the test split.
        random_state (int): Random seed.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Train an XGBoost Regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        XGBRegressor: Trained XGBoost model.
    """
    model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    """
    Train a LightGBM Regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        LGBMRegressor: Trained LightGBM model.
    """
    model = LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series) -> MLPRegressor:
    """
    Train a Multilayer Perceptron (MLP) Regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        MLPRegressor: Trained MLP model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42)
    model.fit(X_scaled, y_train)
    return model