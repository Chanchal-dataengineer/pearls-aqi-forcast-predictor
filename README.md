#  Pearls AQI Predictor

An End-to-End Machine Learning Project for Air Quality Index (AQI) Prediction.

Pearls AQI Predictor uses pollutant data and time-based features to predict Air Quality Index (AQI) using Machine Learning. The project covers the complete Machine Learning workflow from data collection and preprocessing to model training, evaluation, explainability, model registry, and deployment.

---

##  Project Overview

Air quality is an important environmental and public-health concern. This project focuses on building a Machine Learning system that learns patterns from air-pollution data and predicts AQI values.

The project follows an end-to-end Machine Learning pipeline:

**Data Collection → Data Preprocessing → EDA → Feature Engineering → Feature Store → Model Training → Model Evaluation → SHAP Analysis → Model Registry → Deployment**

The project uses air-pollution data collected through the OpenWeather API for Nagarparkar and applies Machine Learning techniques to build an AQI prediction system.

---

##  Project Objectives

The main objectives of this project are:

- Collect real-world air pollution data
- Clean and preprocess the collected data
- Perform Exploratory Data Analysis (EDA)
- Analyze pollutant relationships and outliers
- Create meaningful time-based features
- Build and evaluate regression models
- Select the best-performing model using RMSE
- Generate AQI predictions
- Apply SHAP for model explainability
- Maintain a lightweight local model registry
- Deploy the trained model through a Streamlit application
- Provide a Flask REST API for prediction

---

##  Key Features

-  Real-world AQI data collection
-  Data preprocessing
-  Exploratory Data Analysis
-  Feature Engineering
   Local Feature Store
-  Machine Learning Model Training
-  Model Evaluation
-  SHAP Explainability
-  Model Registry
-  Streamlit Web Application
-  Flask REST API
-  Saved Machine Learning Model
-  Deployment-ready project structure

---

##  Dataset

The project uses air-pollution data collected using the OpenWeather API.

### Location

**Nagarparkar, Sindh, Pakistan**

The dataset contains pollutant measurements including:

- AQI
- CO
- NO
- NO₂
- O₃
- SO₂
- PM2.5
- PM10
- NH₃

Time information is also used to generate additional features.

---

##  Feature Engineering

Time-based features were created from the datetime column.

The following features were generated:

- `hour`
- `day`
- `month`
- `weekday`

Additional analytical features were also created:

- `AQI_change`
- `AQI_rolling_avg`

These features help capture temporal patterns and changes in air quality.

For model training, the features that could introduce target leakage were excluded from the final prediction feature set.

---

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset and identify relationships between AQI and different pollutants.

The EDA includes:

- Dataset structure
- Data types
- Missing-value analysis
- Duplicate analysis
- Descriptive statistics
- Correlation analysis
- AQI distribution
- Pollutant distributions
- Outlier analysis using IQR
- Weekday distribution
- Correlation heatmap
- Visual analysis of pollutant relationships

In this dataset, PM2.5 and PM10 showed strong positive relationships with AQI.

---

##  Machine Learning Models

Two regression models were trained and evaluated:

### 1. Linear Regression

Linear Regression was used as a baseline regression model.

### 2. Random Forest Regressor

Random Forest Regressor was used as a tree-based Machine Learning model for capturing non-linear relationships between pollutant features and AQI.

The best-performing model is selected based on **Root Mean Squared Error (RMSE)**.

---

##  Model Evaluation

The models are evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater importance to larger errors.

### R² Score

Measures how well the model explains the variation in the target variable.

The model training notebook compares the regression models and selects the model with the better RMSE performance.

---

##  Model Explainability — SHAP

SHAP (SHapley Additive exPlanations) is used to understand the contribution of features to model predictions.

The SHAP analysis helps identify which pollutant features have the greatest influence on the model's predictions.

The analysis provides a better understanding of model behavior instead of treating the Machine Learning model as a complete black box.

---

##  Feature Store

A lightweight local Feature Store has been implemented for storing and retrieving the engineered AQI features.

The Feature Store works with the processed feature-engineered dataset and provides a structured way to manage the features used in the Machine Learning workflow.

---

##  Model Registry

A lightweight local Model Registry has been implemented to manage trained model versions.

### Current Model Version

**AQI Regression Model — v1**

The trained model is stored as:

```text
models/aqi_model.pkl