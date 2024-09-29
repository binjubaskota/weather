# Prediction Script: predictor.py

#!/usr/bin/env python
# coding: utf-8

import dill
import numpy as np
from datetime import datetime
import os
import pandas as pd

def load_trained_model(model_dir='trained_model'):
    """
    Loads the trained Random Forest model from the specified directory.

    Parameters:
        model_dir (str): Directory where the model is saved.

    Returns:
        dict: Loaded model data containing 'model', 'metrics', and 'feature_names'.
    """
    model_filepath = os.path.join(model_dir, 'final-model.pkl')
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"The model file was not found at {model_filepath}.")
    with open(model_filepath, 'rb') as file:
        model = dill.load(file)
    return model

def prepare_input_features(input_date: datetime, feature_names: list):
    month = input_date.month
    day = input_date.day
    day_of_week = input_date.weekday()

    if month in [12, 1, 2]:
        season_winter = 1
        season_spring = 0
        season_summer = 0
    elif month in [3, 4, 5]:
        season_winter = 0
        season_spring = 1
        season_summer = 0
    elif month in [6, 7, 8]:
        season_winter = 0
        season_spring = 0
        season_summer = 1
    else:
        season_winter = 0
        season_spring = 0
        season_summer = 0  # Assuming 'Fall' is represented by all seasons being 0

    feature_dict = {
        'Month': month,
        'Day': day,
        'DayOfWeek': day_of_week,
        'Season_Winter': season_winter,
        'Season_Spring': season_spring,
        'Season_Summer': season_summer,
        'ALLSKY_SFC_UVA': 0,
        'ALLSKY_SFC_UVB': 0,
        'ALLSKY_SFC_UV_INDEX': 0,
        'GWETTOP': 0,
        'PS': 0
    }

    input_features = pd.DataFrame([feature_dict])

    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in input_features.columns:
            input_features[feature] = 0

    # Reorder columns to match training
    input_features = input_features[feature_names]

    return input_features.values

def predict_weather_dates(input_dates, model, feature_names, target_columns):
    input_features_list = [prepare_input_features(datetime.strptime(date, "%Y-%m-%d"), feature_names) for date in input_dates]
    input_features = np.vstack(input_features_list)
    predictions = model.predict(input_features)
    predictions_df = pd.DataFrame(predictions, columns=target_columns)
    predictions_df.insert(0, 'Date', input_dates)
    return predictions_df

# Define target columns
target_columns = ['T2M', 'TS', 'QV2M', 'RH2M', 'WS2M', 'WS10M', 'WD10M',
                 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_LW_DWN']

# Example dates
example_dates = [
    '2023-03-15',  # Spring
    '2023-06-21',  # Summer
    '2023-09-10',  # Fall
    '2023-12-25',  # Winter
    '2024-01-01',  # Winter
    '2024-04-18',  # Spring
    '2024-07-04',  # Summer
    '2024-09-28'   # Fall
]

# Load the trained model
try:
    model_data = load_trained_model()
    loaded_model = model_data['model']
    feature_names = model_data['feature_names']
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(e)
    exit(1)
except KeyError as e:
    print(f"Missing key in model data: {e}")
    exit(1)

# Make predictions
try:
    predicted_weather = predict_weather_dates(example_dates, loaded_model, feature_names, target_columns)
    # Display predictions
    print("\nPredicted Weather Parameters:")
    print(predicted_weather)
except Exception as e:
    print(e)
