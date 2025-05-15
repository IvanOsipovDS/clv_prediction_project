# src/data_preparation/engineer_features.py

import pandas as pd

import pandas as pd

CUTOFF_DATE = "2022-01-01"

def load_raw_data():
    purchases = pd.read_csv("data/raw/amazon-purchases.csv")
    survey = pd.read_csv("data/raw/survey.csv")

    purchases["Order Date"] = pd.to_datetime(purchases["Order Date"])
    
    purchases = purchases[purchases["Order Date"] < CUTOFF_DATE]

    return purchases, survey


def clean_data(purchases: pd.DataFrame) -> pd.DataFrame:
    df = purchases.copy()

    df.dropna(subset=["Order Date", "Purchase Price Per Unit", "Quantity", "Survey ResponseID"], inplace=True)

    df["TotalPrice"] = df["Purchase Price Per Unit"] * df["Quantity"]

    return df