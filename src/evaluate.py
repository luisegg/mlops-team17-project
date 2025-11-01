import os
import mlflow
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
from src.train import DataSplitter, FeaturePreprocessor, TARGET

# ======================================================
#  CONFIGURACIÓN DE MLflow Y VARIABLES DE ENTORNO
# ======================================================

# Load environment variables from .env located at project root
# This prevents committing sensitive values to version control
load_dotenv()

# Read tracking configuration from environment variables
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_tracking_username = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set process env for MLflow and clients that read from env vars
os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password

mlflow.set_tracking_uri(uri=mlflow_tracking_uri)


# ======================================================
#  FUNCIONES AUXILIARES
# ======================================================

def load_params():
    """Carga parámetros desde params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def calculate_metrics(y_true, y_pred):
    """Calcula métricas de regresión"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cv = 100.0 * rmse / np.mean(y_true)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "CV": cv}


# ======================================================
#  FUNCIÓN PRINCIPAL DE EVALUACIÓN
# ======================================================

def evaluate_best_model():
    print("\n=== EVALUANDO EL MEJOR MODELO ===")

    # ------------------------------------------------------
    # Cargar parámetros
    # ------------------------------------------------------
    params = load_params()
    data_path = params["preprocess"]["output"]
    test_size = params.get("split", {}).get("test_size", 0.2)

    # ------------------------------------------------------
    # Cargar dataset y dividir
    # ------------------------------------------------------
    df = pd.read_csv(data_path, parse_dates=["date"])
    splitter = DataSplitter(target_column=TARGET)
    X_train, X_test, y_train, y_test = splitter.split_data(df, test_size=test_size)

    # ------------------------------------------------------
    # Preprocesamiento (mismo usado en train.py)
    # ------------------------------------------------------
    preprocessor = FeaturePreprocessor().build_preprocessor()

    # ------------------------------------------------------
    # Buscar el mejor modelo en MLflow
    # ------------------------------------------------------
    experiment_name = "Steel Energy Regression Comparison"
    mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"No se encontró el experimento '{experiment_name}' en MLflow.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.RMSE ASC"],
        max_results=1
    )

    if not runs:
        raise ValueError("⚠️ No se encontró ningún modelo registrado para evaluar.")

    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"

    print(f"✅ Mejor modelo: {best_run.data.tags.get('mlflow.runName', best_run.info.run_id)}")
    print(f"RMSE registrado: {best_run.data.metrics.get('RMSE', 'N/A')}")
    print(f"Modelo URI: {best_model_uri}")

    # ------------------------------------------------------
    # Cargar y evaluar el modelo
    # ------------------------------------------------------
    model = mlflow.sklearn.load_model(best_model_uri)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    print("\n=== RESULTADOS DE EVALUACIÓN ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ------------------------------------------------------
    # Registrar evaluación en nuevo experimento
    # ------------------------------------------------------
    mlflow.set_experiment("Steel Energy Model Evaluation")

    with mlflow.start_run(run_name="Evaluate_Best_Model"):
        mlflow.log_param("evaluated_model_run_id", best_run.info.run_id)
        mlflow.log_metrics(metrics)

        # Guardar resultados locales
        results_df = X_test.copy()
        results_df["y_true"] = y_test.values
        results_df["y_pred"] = y_pred
        results_df["residual"] = results_df["y_true"] - results_df["y_pred"]

        results_path = "data/processed/evaluation_results.csv"
        results_df.to_csv(results_path, index=False)

        # Subir resultados como artefacto
        mlflow.log_artifact(results_path, "evaluation_results")

    print("\n🏁 Evaluación completada y registrada en MLflow.")


# ======================================================
#  EJECUCIÓN DIRECTA
# ======================================================

if __name__ == "__main__":
    evaluate_best_model()
