import os
import pandas as pd

class LocalFeatureStore:

    def __init__(self, store_path="feature_store/features.csv"):
        self.store_path = store_path

    def save_features(self, dataframe):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        dataframe.to_csv(self.store_path, index=False)
        print("✅ Features saved successfully.")

    def load_features(self):
        if os.path.exists(self.store_path):
            print("✅ Features loaded successfully.")
            return pd.read_csv(self.store_path)
        else:
            raise FileNotFoundError("Feature Store is empty.")