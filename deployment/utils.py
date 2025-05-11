import numpy as np
import joblib
from datetime import datetime

# Constants
N_ROWS_PER_DAY = 1782

NUMERICAL_FEATURES = [
    'onpromotion', 'dcoilwtico', 'cluster', 'transactions', 'year', 'month', 'week', 'quarter',
    'day_of_week', 'is_crisis', 'sales_lag_7', 'rolling_mean_7', 'is_weekend', 'is_holiday',
    'promo_last_7_days', 'days_to_holiday'
]

CATEGORICAL_FEATURES = [
    'store_nbr', 'family', 'locale', 'city', 'state', 'holiday_type',
    'store_type', 'promotion_status'
]

DAYS_MAP = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

def preprocess_input(user_df):
    # Map day_of_week
    user_df['day_of_week'] = user_df['day_of_week'].map(DAYS_MAP).astype(int)

    # Apply label encoders
    for col in CATEGORICAL_FEATURES:
        if col != 'day_of_week':
            le = joblib.load(f'mlruns\\424030811349218175\\53eb1701038b4c1797d4afa444dd8cd8\\artifacts\\label_encoders\\label_encoder_{col}.joblib')
            user_df[col] = le.transform(user_df[col].astype(str))

    # Scale numerical features
    scaler = joblib.load("mlruns\\424030811349218175\\53eb1701038b4c1797d4afa444dd8cd8\\artifacts\\scalers\\robust_scaler_numerical.joblib")
    X_num = scaler.transform(user_df[NUMERICAL_FEATURES]).astype(np.float32)

    # Reshape
    X_num = X_num.reshape((1, N_ROWS_PER_DAY, len(NUMERICAL_FEATURES)))
    X_cat = [
        user_df[col].astype(np.int32).to_numpy().reshape((1, N_ROWS_PER_DAY))
        for col in CATEGORICAL_FEATURES
    ]

    return X_cat + [X_num]


def preprocess_output(log_preds):
    """
    Converts model predictions from log scale back to original scale.

    Args:
        log_preds (np.ndarray): Log-scaled predictions (e.g. from model.predict()).

    Returns:
        np.ndarray: Original-scale predictions (after expm1).
    """
    return np.expm1(log_preds)