"""
Quick script to check if model is registered correctly in MLflow
"""
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

client = mlflow.tracking.MlflowClient()

model_name = "steel-energy-cubist"

print(f"Checking model: {model_name}")
print("=" * 50)

try:
    # Get all versions
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        print(f"[X] No model found with name '{model_name}'")
    else:
        print(f"[OK] Found {len(versions)} version(s) of model '{model_name}':")
        for v in versions:
            print(f"\n  Version: {v.version}")
            print(f"  Stage: {v.current_stage}")
            print(f"  Status: {v.status}")
            print(f"  Run ID: {v.run_id}")

    # Check specifically for Production stage
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if prod_versions:
        print(f"\n[OK] Production version found:")
        for v in prod_versions:
            print(f"  Version: {v.version}")
            print(f"  Stage: {v.current_stage}")
    else:
        print(f"\n[X] No Production version found")
        print("   Please transition a version to 'Production' stage")

except Exception as e:
    print(f"[X] Error: {e}")
