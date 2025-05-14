import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_input(input_df):
    input_df = input_df.copy()

    # Load encoders and mappings
    label_encoders = joblib.load('artifacts/label_encoders.pkl')
    target_encodings = joblib.load('artifacts/target_encodings.pkl')
    low_cardinality = list(label_encoders.keys())  # List of low-cardinality cols

    # Label Encoding
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            input_df[col] = encoder.transform(input_df[col])

    # Target Encoding
    for col, mapping in target_encodings.items():
        if col in input_df.columns:
            input_df[col + '_target'] = input_df[col].map(mapping).fillna(0)

    # Drop original low-cardinality and high-cardinality columns
    input_df.drop(columns=low_cardinality + list(target_encodings.keys()), errors='ignore', inplace=True)

    # Ensure numeric types
    input_df = input_df.apply(pd.to_numeric, errors='ignore')

    # Convert 'transferred' to int if present
    if 'transferred' in input_df.columns:
        input_df['transferred'] = input_df['transferred'].astype(int)

    # Ensure all expected columns are present
    expected_columns = [
        'store_nbr', 'onpromotion', 'transferred', 'dcoilwtico', 'cluster', 'transactions', 
        'year', 'month', 'week', 'quarter', 'is_crisis', 'sales_lag_7', 'rolling_mean_7', 
        'is_weekend', 'is_holiday', 'promo_last_7_days', 'days_to_holiday', 'locale', 
        'store_type', 'promotion_status', 'family_target', 'holiday_type_target', 
        'city_target', 'state_target', 'day_of_week_target'
    ]

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # default fill for missing columns

    return input_df[expected_columns]