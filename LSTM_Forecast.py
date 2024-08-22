import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv('energy_demand_sales.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df['Is Holiday'].replace({False: 0, True: 1}, inplace=True)

for column in df:
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(data = df, columns = categorical_cols)
    
    df.replace({True: 1, False: 0}, inplace=True)
    
target_variable = 'Energy Consumption (MWh)'
cols = [col for col in df.columns if col != target_variable] + [target_variable]
df = df[cols]
    
    
original_dates = df.index

consumption_min = df['Energy Consumption (MWh)'].min()
consumption_max = df['Energy Consumption (MWh)'].max()
for column in df:
    cols_to_scale = df.columns[(df.min() != 0) | (df.max() != 1)]
    df[cols_to_scale] = (df[cols_to_scale] - df[cols_to_scale].min()) / (df[cols_to_scale].max() - df[cols_to_scale].min())
   
 
scaled_data = df.values.astype(float)


# Function to create dataset for multivariate
def create_dataset(data, look_back=1):
    X, Y, dates = [], [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), :-1])
        Y.append(data[i + look_back, -1])
        dates.append(original_dates[i + look_back])
    return np.array(X), np.array(Y), np.array(dates)

# Define look_back period and create the dataset
look_back = 100
X, Y, dates = create_dataset(scaled_data, look_back)

# Split the dataset into training and test sets
train_size = int(len(X) * 0.95)
test_size = len(X) - train_size
trainX, trainY = X[:train_size], Y[:train_size]
testX, testY, test_dates = X[train_size:], Y[train_size:], dates[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(trainX, trainY, batch_size=128, epochs=100)

# Make predictions
predictions = model.predict(testX)

# Invert scaling for predictions
predictions_inverse_scaled = predictions * (consumption_max - consumption_min) + consumption_min

# Invert scaling for actual values
actual = testY * (consumption_max - consumption_min) + consumption_min

#print(actual)
# Scaling back the test labels to its original scale
#testY_reshaped = testY.reshape(-1, 1)
#actual = scaler.inverse_transform(np.concatenate((testX.reshape(testX.shape[0], testX.shape[2]), testY_reshaped), axis=1))[:,-1]


# Plotting the results
plt.figure(figsize=(16,8))
plt.plot(test_dates, actual, label='Actual Demand')
plt.plot(test_dates, predictions_inverse_scaled, label='Predicted Demand')
plt.gcf().autofmt_xdate()  # to format x-axis labels to fit into the plot nicely
plt.xlabel('Date')
plt.ylabel('Energy Demand')
plt.legend()
plt.show()

mae = mean_absolute_error(actual, predictions_inverse_scaled)
print(f"Mean Absolute Error (MAE): {mae}")

mse = mean_squared_error(actual, predictions_inverse_scaled)
print(f"Mean Squared Error (MSE): {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

mape = np.mean(np.abs((actual - predictions_inverse_scaled) / actual)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")