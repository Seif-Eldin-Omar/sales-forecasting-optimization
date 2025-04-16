import pandas as pd

def preprocess(df):
    """
    Preprocess the input DataFrame by handling missing values, optimizing data types,
    and engineering relevant features for forecasting and modeling.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    # =====================
    # Missing Value Handling
    # =====================
    df['holiday_type'].fillna('Normal Day', inplace=True)
    df['locale'].fillna('None', inplace=True)
    df['locale_name'].fillna('None', inplace=True)
    df['description'].fillna('None', inplace=True)
    df['transferred'].fillna(False, inplace=True)
    df['dcoilwtico'].fillna(method='ffill', inplace=True)
    df['dcoilwtico'].fillna(method='bfill', inplace=True)
    df['transactions'].fillna(0, inplace=True)

    # =====================
    # Feature Engineering
    # =====================

    # Crisis Indicator
    df['is_crisis'] = df['description'].apply(
        lambda x: 1 if isinstance(x, str) and x.startswith('Terremoto Manabi') else 0
    ).astype('int8')

    # Sales Lag
    df['sales_lag_7'] = (
        df.groupby(['store_nbr', 'family'])['sales']
        .shift(7)
    )

    # Rolling Mean (Previous 7 Days)
    df['rolling_mean_7'] = (
        df.groupby(['store_nbr', 'family'])['sales']
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # Is Weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')

    # Is Holiday
    df['is_holiday'] = (df['holiday_type'] != 'Normal Day').astype('int8')

    # Promo History (last 7 days)
    df['promo_last_7_days'] = (
        df.groupby(['store_nbr', 'family'])['onpromotion']
        .shift(1)
        .rolling(window=7, min_periods=1)
        .sum()
    )

    # =====================
    # Fill Feature-Engineered NaNs
    # =====================
    df['sales_lag_7'].fillna(method='bfill', inplace=True)
    df['rolling_mean_7'].fillna(method='bfill', inplace=True)
    df['promo_last_7_days'].fillna(0, inplace=True)  # assume no promos in early data

    # =====================
    # Data Type Optimization
    # =====================
    df['week'] = pd.to_numeric(df['week'], errors='coerce').astype('Int8')
    df['onpromotion'] = df['onpromotion'].astype('int8')
    df['is_crisis'] = df['is_crisis'].astype('int8')  # this will be redefined later
    df['is_weekend'] = df['is_weekend'].astype('int8', errors='ignore')  # may be redefined
    df['is_holiday'] = df['is_holiday'].astype('int8', errors='ignore')  # may be redefined
    df['store_nbr'] = df['store_nbr'].astype('int16')
    df['cluster'] = df['cluster'].astype('int8')

    
    return df