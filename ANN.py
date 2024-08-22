import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the dataset
data = pd.read_csv('energy_demand_sales.csv')

# Ensure 'Date' is datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# One-hot encode categorical variables (e.g., 'Day of the Week')
data = pd.get_dummies(data, columns=['Day of the Week'], drop_first=True)

# Assuming 'Energy Consumption (MWh)' is the target
X = data.drop(['Energy Consumption (MWh)', 'Date'], axis=1)  # Drop 'Date' if it's not used as a feature
y = data['Energy Consumption (MWh)']
dates = data['Date']  # Keep dates for plotting

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
dates_train, dates_test = train_test_split(dates, test_size=0.1, random_state=42, shuffle=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(), loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=1000, verbose=1)  # Set verbose to 1 to see training progress

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# Plotting Actual vs Predicted Values
plt.figure(figsize=(14, 7))
plt.plot(dates_test, y_test, label='Actual', marker='o', linestyle='-', alpha=0.7)
plt.plot(dates_test, y_pred, label='Predicted', linestyle='--', alpha=0.7)
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation Metrics
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape}%")

