import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.filterwarnings("ignore")

df = pd.read_csv('data/cleaned_data.csv', parse_dates=['date'])
print(df.columns)
print(df.dtypes)

df['locale'].fillna('None', inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
print(df.dtypes)

train_df = df[df['date'] < '2017-01-01']
test_df = df[df['date'] >= '2017-01-01']

y_train = np.log1p(train_df['sales'])
y_test = np.log1p(test_df['sales'])

train_df_copy = train_df.copy()
test_df_copy = test_df.copy()

categorical_cols = train_df_copy.select_dtypes(include=['object', 'category']).columns
low_cardinality = [col for col in categorical_cols if train_df_copy[col].nunique() <= 5]
high_cardinality = [col for col in categorical_cols if train_df_copy[col].nunique() > 5]

print("Low Cardinality Columns:", low_cardinality)
print("High Cardinality Columns:", high_cardinality)
print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", train_df_copy.select_dtypes(exclude=['object', 'category']).columns)


train_df_encoded = pd.DataFrame(index=train_df_copy.index)
test_df_encoded = pd.DataFrame(index=test_df_copy.index)

for col in low_cardinality:
    le = LabelEncoder()
    train_df_encoded[col] = le.fit_transform(train_df_copy[col])
    test_df_encoded[col] = le.transform(test_df_copy[col])

for col in high_cardinality:
    train_mean_target = train_df_copy.groupby(col)['sales'].mean()
    train_df_encoded[col + '_target'] = train_df_copy[col].map(train_mean_target)
    test_df_encoded[col + '_target'] = test_df_copy[col].map(train_mean_target)

train_df_final = train_df_copy.drop(columns=low_cardinality + high_cardinality)
test_df_final = test_df_copy.drop(columns=low_cardinality + high_cardinality)

train_df_final = pd.concat([train_df_final, train_df_encoded], axis=1)
test_df_final = pd.concat([test_df_final, test_df_encoded], axis=1)

train_df_final = train_df_final.apply(pd.to_numeric, errors='ignore')
test_df_final = test_df_final.apply(pd.to_numeric, errors='ignore')

drop_cols = ['date', 'sales']
X_train = train_df_final.drop(columns=drop_cols)
X_test = test_df_final.drop(columns=drop_cols)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train.dtypes)
print(X_test.dtypes)
print(X_train.select_dtypes(include=['object']).columns)
print(X_test.select_dtypes(include=['object']).columns)
print(X_train.isnull().sum())
print(X_test.isnull().sum())

X_train['transferred'] = X_train['transferred'].astype(int)
X_test['transferred'] = X_test['transferred'].astype(int)

model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, 'models/xgb_model.pkl')

y_pred = model.predict(X_test)
y_pred = np.expm1(y_pred)
y_test_real = np.expm1(y_test)

mse = mean_squared_error(y_test_real, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_real, y_pred)

print(f'XGBoost \nMSE: {mse:.2f} \nRMSE: {rmse:.2f} \nR2: {r2:.2f}')

xgb.plot_importance(model, max_num_features=15)
plt.tight_layout()
plt.show()
