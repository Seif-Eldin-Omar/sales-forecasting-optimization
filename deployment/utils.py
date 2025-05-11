import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_input(input_df, reference_df=None):
    input_df = input_df.copy()

    # If reference_df is not provided, load the cleaned_data.csv dynamically
    if reference_df is None:
        reference_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data.csv')
        reference_df = pd.read_csv(reference_path, parse_dates=['date'])

    # Handle missing values for 'locale', 'store_type', and 'promotion_status'
    input_df['locale'].fillna('None', inplace=True)
    reference_df['locale'].fillna('None', inplace=True)

    input_df['store_type'].fillna('None', inplace=True)
    reference_df['store_type'].fillna('None', inplace=True)

    input_df['promotion_status'].fillna('Inactive', inplace=True)
    reference_df['promotion_status'].fillna('Inactive', inplace=True)

    # Identify categorical columns
    categorical_cols = reference_df.select_dtypes(include=['object', 'category']).columns
    low_cardinality = [col for col in categorical_cols if reference_df[col].nunique() <= 5]
    high_cardinality = [col for col in categorical_cols if reference_df[col].nunique() > 5]

    # Encode low cardinality with LabelEncoder
    for col in low_cardinality:
        le = LabelEncoder()
        le.fit(reference_df[col])
        # Handle unseen labels by defaulting to -1
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Target encoding for high cardinality
    for col in high_cardinality:
        # Check if the column exists in the input data
        if col in input_df.columns:
            target_map = reference_df.groupby(col)['sales'].mean()
            input_df[col + '_target'] = input_df[col].map(target_map)
            input_df.drop(columns=[col], inplace=True)
        else:
            print(f"Warning: Column '{col}' not found in input data. Skipping target encoding for this column.")

    # Drop the original low cardinality columns after encoding
    input_df.drop(columns=low_cardinality, inplace=True)

    # Convert data to numeric
    input_df = input_df.apply(pd.to_numeric, errors='ignore')

    # Cast specific columns like 'transferred' if necessary
    if 'transferred' in input_df.columns:
        input_df['transferred'] = input_df['transferred'].astype(int)

    # Reorder columns to match the model's expected feature order
    expected_columns = [
        'store_nbr', 'onpromotion', 'transferred', 'dcoilwtico', 'cluster', 'transactions', 
        'year', 'month', 'week', 'quarter', 'is_crisis', 'sales_lag_7', 'rolling_mean_7', 
        'is_weekend', 'is_holiday', 'promo_last_7_days', 'days_to_holiday', 'locale', 
        'store_type', 'promotion_status', 'family_target', 'holiday_type_target', 
        'city_target', 'state_target', 'day_of_week_target'
    ]

    # Ensure input_df contains all the required columns (add missing ones if necessary)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill with zeros or a default value

    input_df = input_df[expected_columns]  # Reorder columns to match the model

    return input_df