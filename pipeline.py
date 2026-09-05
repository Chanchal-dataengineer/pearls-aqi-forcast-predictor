import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


SCRIPTS = [
    "scripts/collect_data.py",
    "scripts/preprocess_data.py",
    "scripts/feature_engineering.py",
    "scripts/train_model.py",
]


def run_step(script):

    print()
    print("=" * 60)
    print(f"RUNNING: {script}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print()
        print(f"PIPELINE FAILED: {script}")
        sys.exit(result.returncode)

    print()
    print(f"COMPLETED: {script}")


def main():

    print()
    print("=" * 60)
    print("PEARLS AQI PREDICTOR")
    print("AUTOMATED ML PIPELINE")
    print("=" * 60)

    for script in SCRIPTS:
        run_step(script)

    print()
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()