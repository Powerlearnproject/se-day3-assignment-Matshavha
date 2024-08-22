import streamlit as st
import pandas as pd
import joblib

# Load your trained model pipeline
model = joblib.load('gradient_boosting_model.joblib')

# Title for your app
st.title('Energy Consumption Prediction')

# Creating user input fields
year = st.number_input('Year', min_value=1990, max_value=2030, value=2023)
month = st.number_input('Month', min_value=1, max_value=12, value=1)
day = st.number_input('Day', min_value=1, max_value=31, value=1)
day_of_week = st.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
is_holiday = st.selectbox('Is Holiday', ['No', 'Yes'])
pv_penetration = st.number_input('PV Penetration (%)')
gdp_growth_rate = st.number_input('GDP Growth Rate (%)')
population = st.number_input('Population', value=100000)
loadshedding_stage = st.number_input('Loadshedding Stage', value=0)
ev_penetration = st.number_input('EV Penetration (Number)')
temperature = st.number_input('Temperature (°C)')
humidity = st.number_input('Humidity (%)')

# Button to make prediction
if st.button('Predict'):
    # Organize the inputs into the same structure as your training data
    input_data = {
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Day of the Week': [day_of_week],
        'Is Holiday': [1 if is_holiday == 'Yes' else 0],
        'PV Penetration (%)': [pv_penetration],
        'GDP Growth Rate (%)': [gdp_growth_rate],
        'Population': [population],
        'Loadshedding Stage': [loadshedding_stage],
        'EV Penetration (Number)': [ev_penetration],
        'Temperature (°C)': [temperature],
        'Humidity (%)': [humidity],
    }
    input_df = pd.DataFrame.from_dict(input_data)

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display the prediction
    st.success(f'Predicted Energy Consumption: {prediction}')

