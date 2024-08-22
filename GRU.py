import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load your dataset
file_path = 'energy_demand_sales.csv'
data = pd.read_csv(file_path)

# Ensure 'Date' is datetime type and sort the data by date
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Extract temporal features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek
dates = data['Date']  # Save dates for plotting
data.drop(['Date'], axis=1, inplace=True)

# One-hot encode categorical variables and drop the original columns if needed
data = pd.get_dummies(data)

# Specify the target column name
target_column = 'Energy Consumption (MWh)'

# Initialize separate scalers for features and target
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale features and target separately
scaled_features = feature_scaler.fit_transform(data.drop([target_column], axis=1))
scaled_target = target_scaler.fit_transform(data[[target_column]].values)

# Concatenate scaled features and target to form the scaled dataset
scaled_data = np.concatenate([scaled_features, scaled_target], axis=1)
target_index = data.columns.get_loc(target_column)

# Define a function to create sequences for multiple features
def create_multifeature_sequences(data, target_index, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, target_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 14
X, y = create_multifeature_sequences(scaled_data, -1, seq_length)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
dates_train, dates_test = train_test_split(dates[seq_length+1:], test_size=0.1, random_state=42, shuffle=False)

# Define the GRU model
model = Sequential([
    GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Predict
predicted = model.predict(X_test)

# Correctly apply inverse transform to predictions and y_test using the target scaler
predicted_inverse = target_scaler.inverse_transform(predicted)
y_test_inverse = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics
mse = mean_squared_error(y_test_inverse, predicted_inverse)
mae = mean_absolute_error(y_test_inverse, predicted_inverse)
mape = np.mean(np.abs((y_test_inverse - predicted_inverse) / y_test_inverse)) * 100

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Absolute Percentage Error: {mape}%')

# Plotting the results with dates
plt.figure(figsize=(14, 7))
plt.plot(dates_test.reset_index(drop=True), y_test_inverse, label='Actual', linestyle='-', markersize=5, alpha=0.7)
plt.plot(dates_test.reset_index(drop=True), predicted_inverse, label='Predicted', linestyle='--', markersize=5, alpha=0.7)
plt.title('Energy Consumption Prediction')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MWh)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

