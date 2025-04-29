# src/data_preparation/engineer_features.py

import pandas as pd

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw transaction data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded raw data as a pandas DataFrame.
    """
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transaction data:
    - Remove duplicates
    - Handle missing values
    - Fix column types if needed

    Args:
        df (pd.DataFrame): Raw transaction data.

    Returns:
        pd.DataFrame: Cleaned transaction data.
    """
    # Drop duplicate rows
    df_cleaned = df.drop_duplicates()

    # Fill missing values if necessary (placeholder)
    # df_cleaned = df_cleaned.fillna(method='ffill')

    return df_cleaned