import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Load the dataset
file_path = 'energy_demand_sales.csv'
data = pd.read_csv(file_path)

dates = data['Date']  # Store the dates for later plotting

# Preprocessing
# Convert 'Date' to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.drop(['Date'], axis=1, inplace=True)

# Handling categorical variables with OneHotEncoder
categorical_features = ['Day of the Week', 'Is Holiday']
one_hot_encoder = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features)],
    remainder='passthrough')

# Split the dataset
X = data.drop('Energy Consumption (MWh)', axis=1)
y = data['Energy Consumption (MWh)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.009, random_state=42)

test_dates = dates.iloc[X_test.index]  # Extract test dates using the test set indices

test_dates_datetime = pd.to_datetime(test_dates)

# Define and train the Gradient Boosting model
model = make_pipeline(preprocessor, GradientBoostingRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

joblib.dump(model, 'gradient_boosting_model.joblib')

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape}%")

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test, label='Actual', alpha=0.7)
plt.plot(test_dates, y_pred, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.xticks(rotation=45)
plt.legend()
plt.show()


