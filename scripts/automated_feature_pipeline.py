import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def run_script(script_name):
    script_path = BASE_DIR / "scripts" / script_name

    print("\n" + "=" * 60)
    print(f"Running: {script_name}")
    print("=" * 60)

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed.")


def main():
    print("\n🌍 AQI Automated Feature Pipeline")
    print("=" * 60)

    # 1. Collect latest AQI data
    run_script("collect_data.py")

    # 2. Clean and preprocess data
    run_script("preprocess_data.py")

    # 3. Generate ML features
    run_script("feature_engineering.py")

    print("\n" + "=" * 60)
    print("✅ Automated feature pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()