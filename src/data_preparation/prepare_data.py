# src/data_preparation/engineer_features.py

import pandas as pd

def load_raw_data() -> pd.DataFrame:
    purchases = pd.read_csv('data/raw/amazon-purchases.csv')
    survey = pd.read_csv('data/raw/survey.csv')

    purchases['Order date'] = pd.to_datetime(purchases['Order date'])
    purchases['CustomerID'] = purchases['Survey ResponseID']
    purchases['TotalPrice'] = purchases['Purchase price per unit'] * purchases['Quantity']

    df = purchases.merge(survey, on='Survey ResponseID', how='left')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Простейшая очистка, можно дополнить
    return df.dropna(subset=['CustomerID', 'Order date', 'TotalPrice'])