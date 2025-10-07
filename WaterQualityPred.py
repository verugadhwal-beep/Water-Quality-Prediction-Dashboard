import pandas as pd # data manipulation
import numpy as np # numerical python - linear algebra

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
print(df.head()) 

df.info()

print(df.shape)

print(df.describe().T)

print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
print(df.head())

df.info()

df = df.sort_values(by=['id', 'date'])
print(df.head())

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
print(df.head())

print(df.columns)

pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4',
  'PO4', 'CL']
print(df.head())

#WEEK 2

df = df.dropna(subset=pollutants)
print(df.head())

print(df.isnull().sum())


X = df[['id', 'year']]
y = df[pollutants]

X_encoded = pd.get_dummies(X, columns=['id'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=43)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=43))
print(model.fit(X_train, y_train))

y_pred = model.predict(X_test)

print("Model Performance on the Test Data:")
for i, pollutant in enumerate(pollutants):
    print(f'{pollutant}:')
    print('   MSE:', mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print('   R2:', r2_score(y_test.iloc[:, i], y_pred[:, i]))
    print()

    station_id = '22'
year_input = 2024

input_data = pd.DataFrame({'year': [year_input], 'id': [station_id]})
input_encoded = pd.get_dummies(input_data, columns=['id'])

missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]  

predicted_pollutants = model.predict(input_encoded)[0]

print(f"\nPredicted pollutant levels for station '{station_id}' in {year_input}:")
for p, val in zip(pollutants, predicted_pollutants):
    print(f"  {p}: {val:.2f}")

import joblib

joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X_encoded.columns.tolist(), "model.pkl")
print('Model and cols structure are saved!')