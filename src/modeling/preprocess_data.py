# src/modeling/preprocess_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_modeling_data(path: str = "data/processed/rfm_clv.csv", target_col: str = "LogFutureCLV"):
    """
    Load processed dataset and split into features and target.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)