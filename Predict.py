import pandas as pd
import joblib

def make_prediction():
    # Load the saved pipeline (including preprocessing and model)
    pipeline = joblib.load('gradient_boosting_model.joblib')
    
    print("Please enter feature values for prediction:")
    # Directly collect input for "Day of the Week" as it was originally before encoding
    input_features = {
        'Year': int(input("Year (e.g., 2023): ")),
        'Month': int(input("Month (1-12): ")),
        'Day': int(input("Day (1-31): ")),
        'Day of the Week': input("Day of the Week (e.g., 'Monday'): "),
        'Is Holiday': int(input("Is Holiday (1 for Yes, 0 for No): ")),
        'PV Penetration (%)': float(input("PV Penetration (%): ")),
        'GDP Growth Rate (%)': float(input("GDP Growth Rate (%): ")),
        'Population': int(input("Population: ")),
        'Loadshedding Stage': int(input("Loadshedding Stage: ")),
        'EV Penetration (Number)': int(input("EV Penetration (Number): ")),
        'Temperature (°C)': float(input("Temperature (°C): ")),
        'Humidity (%)': float(input("Humidity (%): "))
    }
    
    # Construct a DataFrame from input_features
    input_df = pd.DataFrame([input_features])
    
    # Ensure your 'Day of the Week' input matches how it was handled/trained in the model
    # For example, if it was one-hot encoded during training, the pipeline should automatically
    # take care of transforming this feature based on how the ColumnTransformer was set up
    
    # Predict using the loaded pipeline
    predicted_energy = pipeline.predict(input_df)
    
    return predicted_energy[0]

# Example usage of the function
if __name__ == "__main__":
    prediction = make_prediction()
    print(f"Predicted Energy Consumption: {prediction}")
