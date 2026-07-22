import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

print("Loading latest features...")

df = pd.read_csv("feature_store/features.csv")

X = df.drop("AQI", axis=1)
y = df["AQI"]

print("Training latest model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(
    model,
    "models/random_forest_model_latest.pkl"
)

print("✅ Latest model trained successfully.")