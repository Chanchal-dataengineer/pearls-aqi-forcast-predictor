import os
import shutil

class ModelRegistry:

    def __init__(self, registry_path="model_registry"):
        self.registry_path = registry_path
        os.makedirs(self.registry_path, exist_ok=True)

    def register_model(self, model_path, version="v1"):
        destination = os.path.join(
            self.registry_path,
            f"random_forest_{version}.pkl"
        )

        shutil.copy(model_path, destination)

        print(f"✅ Model registered successfully: {destination}")

        return destination