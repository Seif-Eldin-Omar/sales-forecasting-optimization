import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from utils import preprocess_input

# Load model with absolute path
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_model.pkl')
model = joblib.load(model_path)

st.title("Sales Forecasting using XGBoost")



store_nbr = st.number_input("Store Number", min_value=1, max_value=100)

family = st.selectbox("Product Family", ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES'])

onpromotion = st.number_input("On Promotion", min_value=0, max_value=100)

locale = st.selectbox("Locale", ['None', 'Local', 'Regional', 'National'])

transferred = st.selectbox("Transferred", [0, 1])

dcoilwtico = st.number_input("Crude Oil Price", min_value=0.0)

city = st.selectbox("City", ['Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
        'Santo Domingo', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil',
        'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad',
        'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen'])  

state = st.selectbox("State", ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
        'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
        'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
        'El Oro', 'Esmeraldas', 'Manabi'])  

store_type = st.selectbox("Store Type", ['A', 'B', 'C', 'D', 'E'])

cluster = st.number_input("Cluster", min_value=1, max_value=20)

transactions = st.number_input("Transactions", min_value=0)

year = st.number_input("Year", min_value=2013, max_value=2030)

month = st.number_input("Month", min_value=1, max_value=12)

week = st.number_input("Week", min_value=1, max_value=53)

quarter = st.number_input("Quarter", min_value=1, max_value=4)

day_of_week = st.number_input("Day of Week", min_value=0, max_value=6)

is_crisis = st.selectbox("Is Crisis", [0, 1])

sales_lag_7 = st.number_input("Sales Lag 7", min_value=0.0)

rolling_mean_7 = st.number_input("Rolling Mean 7", min_value=0.0)

is_weekend = st.selectbox("Is Weekend", [0, 1])

is_holiday = st.selectbox("Is Holiday", [0, 1])

promo_last_7_days = st.number_input("Promo Last 7 Days", min_value=0)

days_to_holiday = st.number_input("Days to Holiday", min_value=0)

promotion_status = st.selectbox("Promotion Status", ['Active', 'Inactive'])

holiday_type = st.selectbox("Holiday Type", ['Work Day', 'Transfer', 'Additional', 'Holiday', 'Bridge', 'Event', 'None'])

if st.button("Predict Sales"):
    input_data = pd.DataFrame({
        'store_nbr': [store_nbr],
        'family': [family],
        'onpromotion': [onpromotion],
        'locale': [locale],
        'transferred': [transferred],
        'dcoilwtico': [dcoilwtico],
        'city': [city],
        'state': [state],
        'holiday_type': [holiday_type],  
        'store_type': [store_type],
        'cluster': [cluster],
        'transactions': [transactions],
        'year': [year],
        'month': [month],
        'week': [week],
        'quarter': [quarter],
        'day_of_week': [day_of_week],  
        'is_crisis': [is_crisis],
        'sales_lag_7': [sales_lag_7],
        'rolling_mean_7': [rolling_mean_7],
        'is_weekend': [is_weekend],
        'is_holiday': [is_holiday],
        'promo_last_7_days': [promo_last_7_days],
        'days_to_holiday': [days_to_holiday],
        'promotion_status': [promotion_status]
    })

    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    prediction = np.expm1(prediction)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")
