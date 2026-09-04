# Pearls AQI Predictor

## Project Overview

Pearls AQI Predictor is an end-to-end Machine Learning system for Air Quality Index prediction.

The project covers:

- Historical AQI data collection
- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Feature storage
- Machine learning model training
- Model evaluation
- Model versioning
- SHAP explainability
- Streamlit dashboard
- Flask REST API
- GitHub Actions validation

## Objective

The objective is to build a reproducible AQI prediction pipeline that uses pollutant measurements and temporal features to estimate future AQI values.

## Data Source

Air quality data is collected through the OpenWeather Air Pollution API.

The project uses pollutant variables including:

- CO
- NO
- NO2
- O3
- SO2
- PM2.5
- PM10
- NH3

## Feature Engineering

The feature engineering pipeline creates:

- Hour
- Day
- Month
- Weekday
- AQI lag
- AQI change
- Rolling AQI average

These features provide temporal information to the machine learning model.

## Machine Learning

The training pipeline experiments with regression models and evaluates them using:

- MAE
- RMSE
- R²

The final selected model is stored as:

`models/aqi_model.pkl`

The corresponding feature schema is stored as:

`models/feature_columns.pkl`

## Model Registry

A lightweight local model registry is implemented to maintain model versions and metadata.

Registered artifacts are stored under:

`model_registry/`

## Explainability

SHAP is used to understand feature contributions and identify which variables have the greatest influence on model predictions.

## Dashboard

The Streamlit dashboard provides:

- Pollutant input
- Current AQI input
- Prediction date and time
- Three-step recursive AQI forecast
- Forecast visualization
- Model information
- Feature inspection

Run:

```bash
streamlit run app.py