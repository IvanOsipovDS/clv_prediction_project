# src/feature_engineering/engineer_features.py

import pandas as pd

from src.data_preparation import prepare_data

cutoff_date = "2022-01-01"

def create_rfm_features(df: pd.DataFrame, customer_id_col: str, invoice_date_col: str, amount_col: str) -> pd.DataFrame:
    """
    Create RFM (Recency, Frequency, Monetary) features for each customer.

    Args:
        df (pd.DataFrame): Cleaned transaction data.
        customer_id_col (str): Name of the customer ID column.
        invoice_date_col (str): Name of the transaction date column.
        amount_col (str): Name of the transaction amount column.

    Returns:
        pd.DataFrame: DataFrame with RFM and additional features.
    """
    # Set reference date as the maximum invoice date in the dataset
    reference_date = df[invoice_date_col].max()

    # Group transactions by customer
    customer_group = df.groupby(customer_id_col)

    # Recency: Number of days since last purchase
    recency = customer_group[invoice_date_col].max().apply(lambda x: (reference_date - x).days)

    # Frequency: Number of purchases
    frequency = customer_group[invoice_date_col].count()

    # Monetary: Total amount spent
    monetary = customer_group[amount_col].sum()

    # Average purchase value
    avg_purchase_value = monetary / frequency

    # Customer lifespan (days): from first purchase to reference date
    lifespan = customer_group[invoice_date_col].min().apply(lambda x: (reference_date - x).days)

    # Mean time between purchases (Interpurchase Time)
    mean_days_between_purchases = lifespan / frequency

    clv_12m = df[
            (df[invoice_date_col] < cutoff_date + pd.DateOffset(months=12))
        ].groupby(customer_id_col)[amount_col].sum()

    # Create a DataFrame with all features
    features_df = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'AveragePurchaseValue': avg_purchase_value,
        'CustomerLifespan': lifespan,
        'MeanDaysBetweenPurchases': mean_days_between_purchases
    })

    return features_df

def create_clv_targets(df: pd.DataFrame, customer_id_col: str, invoice_date_col: str, amount_col: str,
                        cutoff_date: pd.Timestamp, window_months: int = None) -> pd.DataFrame:
    """
    Create CLV targets for a given cutoff date.

    Args:
        df (pd.DataFrame): Cleaned transaction data.
        customer_id_col (str): Name of the customer ID column.
        invoice_date_col (str): Name of the transaction date column.
        amount_col (str): Name of the transaction amount column.
        cutoff_date (pd.Timestamp): Date to split past and future.
        window_months (int, optional): If specified, computes CLV only for a fixed period (e.g., 12 months).

    Returns:
        pd.DataFrame: DataFrame with CLV targets (and log-transformed versions).
    """
    df_future = df[df[invoice_date_col] >= cutoff_date].copy()

    if window_months:
        df_future = df_future[df_future[invoice_date_col] < cutoff_date + pd.DateOffset(months=window_months)]

    clv = df_future.groupby(customer_id_col)[amount_col].sum().rename('FutureCLV')
    clv_df = clv.to_frame()
    clv_df['LogFutureCLV'] = clv_df['FutureCLV'].apply(lambda x: pd.np.log1p(x))

    return clv_df.reset_index()

purchases, survey = prepare_data.load_raw_data()
df = prepare_data.clean_data(purchases)

rfm_df = create_rfm_features(
    df,
    customer_id_col='Survey ResponseID',
    invoice_date_col='Order Date',
    amount_col='TotalPrice'
)