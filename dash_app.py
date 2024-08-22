import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('gradient_boosting_model.joblib')

# Initialize the Dash app
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# App layout
app.layout = html.Div(children=[
    html.H1(children='Energy Consumption Prediction', style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),

    html.Div(children=[
        dcc.Input(id='Year', type='number', placeholder='Year', min=2000, max=2030, step=1, className='input-field'),
        dcc.Input(id='Month', type='number', placeholder='Month', min=1, max=12, step=1, className='input-field'),
        dcc.Input(id='Day', type='number', placeholder='Day', min=1, max=31, step=1, className='input-field'),
        dcc.Dropdown(id='Day of the Week', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], placeholder='Day of the Week', className='dropdown'),
        dcc.Input(id='Is Holiday', type='number', placeholder='Is Holiday (1 for Yes, 0 for No)', min=0, max=1, step=1, className='input-field'),
        dcc.Input(id='PV Penetration (%)', type='number', placeholder='PV Penetration (%)', className='input-field'),
        dcc.Input(id='GDP Growth Rate (%)', type='number', placeholder='GDP Growth Rate (%)', className='input-field'),
        dcc.Input(id='Population', type='number', placeholder='Population', className='input-field'),
        dcc.Input(id='Loadshedding Stage', type='number', placeholder='Loadshedding Stage', className='input-field'),
        dcc.Input(id='EV Penetration (Number)', type='number', placeholder='EV Penetration (Number)', className='input-field'),
        dcc.Input(id='Temperature (°C)', type='number', placeholder='Temperature (°C)', className='input-field'),
        dcc.Input(id='Humidity (%)', type='number', placeholder='Humidity (%)', className='input-field'),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginTop': 20}),

    html.Button(id='submit-button', n_clicks=0, children='Predict', style={'marginTop': 20, 'width': '200px', 'height': '40px', 'backgroundColor': '#007BFF', 'color': 'white', 'fontSize': '20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),

    html.Div(id='prediction-output', style={'marginTop': 20, 'textAlign': 'center', 'fontSize': '24px'})
])

# Callback for updating prediction output
@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [Input('Year', 'value'), Input('Month', 'value'), Input('Day', 'value'),
     Input('Day of the Week', 'value'), Input('Is Holiday', 'value'),
     Input('PV Penetration (%)', 'value'), Input('GDP Growth Rate (%)', 'value'),
     Input('Population', 'value'), Input('Loadshedding Stage', 'value'),
     Input('EV Penetration (Number)', 'value'), Input('Temperature (°C)', 'value'),
     Input('Humidity (%)', 'value')]
)
def update_output(n_clicks, Year, Month, Day, Day_of_the_Week, Is_Holiday, PV_Penetration, GDP_Growth_Rate, Population, Loadshedding_Stage, EV_Penetration, Temperature, Humidity):
    if n_clicks > 0:
        input_data = pd.DataFrame([[Year, Month, Day, Day_of_the_Week, Is_Holiday, PV_Penetration, GDP_Growth_Rate, Population, Loadshedding_Stage, EV_Penetration, Temperature, Humidity]],
                                  columns=['Year', 'Month', 'Day', 'Day of the Week', 'Is Holiday', 'PV Penetration (%)', 'GDP Growth Rate (%)', 'Population', 'Loadshedding Stage', 'EV Penetration (Number)', 'Temperature (°C)', 'Humidity (%)'])
        prediction = model.predict(input_data)[0]
        return f'Predicted Energy Consumption: {prediction}'
    else:
        return ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
