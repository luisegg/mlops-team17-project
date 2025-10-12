from errno import ESTALE
import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from cubist import Cubist
import numpy as np
import mlflow
from mlflow.models import infer_signature
import joblib
import json
import os
from dotenv import load_dotenv

TARGET = "Usage_kWh"

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

def load_params(path="params.yaml"):
    if Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError("params.yaml no encontrado")

def log_run_to_mlflow(experiment_name, params, metrics, model=None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        #if model is not None:
            #mlflow.sklearn.log_model(model, "model")

def calculate_metrics(y_true, y_pred):
    """
    Calculate RMSE, MAE, and R2 metrics for regression models.
    Also prints the metrics to console.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing RMSE, MAE, and R2 metrics
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cv = cv_percent(y_true, y_pred)
    
    # Print metrics
    #print('RMSE:', rmse)
    #print('MAE:', mae)
    #print('R2:', r2)
    #print('CV', cv)
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "CV": cv}

def cv_percent(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    return 100.0 * rmse / np.mean(y_true)

def split_train_test_data(df, test_size):
    # Separates X/y
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    # Split with scikit-learn
    # order by date and NO shuffle (temporal)
    order = df["date"].argsort()
    X = X.iloc[order]
    y = y.iloc[order]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    print(f"[split] test_size={test_size}")
    print(f"[split] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[split] y_train: {y_train.shape} | y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def build_preprocessor():
    pre = ColumnTransformer(
        transformers=[
            ('numeric_cols', Pipeline([
                ('Standard_Scaler', StandardScaler())
            ]), ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']),
            ('Day_of_week_oh', Pipeline([
                ('ohe', OneHotEncoder(
                categories=[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']],
                drop='first',
                sparse_output=False))
            ]), ['Day_of_week']),
            ('WeekStatus_bi', Pipeline([ # produce 1 col: Weekend=0, Weekday=1
                ('bin', OneHotEncoder(
                    categories=[['Weekend', 'Weekday']],
                    drop='first',
                    sparse_output=False
                ))
            ]), ['WeekStatus']),
            ('LoadType_ord', Pipeline([
                ('ord', OrdinalEncoder(
                    categories=[['Light_Load','Medium_Load','Maximum_Load']],
                    dtype=np.int8,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ]), ['Load_Type'])
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return pre

def run_linear_regression(X_train, y_train, X_test, y_test, pre):
    """LinearRegression no tiene hiperparámetros: un solo run."""
    exp = "LinearRegression"
    est = LinearRegression(fit_intercept=True)
    pipe = Pipeline([("prep", pre), ("est", est)])

    # Fit the model
    pipe.fit(X_train, y_train)

    #Predicts
    y_pred = pipe.predict(X_test)

    # Gets metrics
    metrics = calculate_metrics(y_test, y_pred)

    log_run_to_mlflow(exp, params={}, metrics=metrics, model=pipe)

def run_knn_regression(X_train, y_train, X_test, y_test, pre, knn_params):
    exp = "KNNRegression"
   
    k_list = knn_params.get("k", [3, 5, 7, 11])
    weights_list = knn_params.get("weights", ["uniform", "distance"])
    p_list = knn_params.get("p", [1, 2])
    metric = "minkowski"

    for k in k_list:
        for w in weights_list:
            for p in p_list:
                params = {"n_neighbors": k, "weights": w, "metric": metric, "p": p}
                est = KNeighborsRegressor(**params)
                pipe = Pipeline([("prep", pre), ("est", est)])

                # Fit the model
                pipe.fit(X_train, y_train)

                #Predicts
                y_pred = pipe.predict(X_test)

                # Gets metrics
                metrics = calculate_metrics(y_test, y_pred)

                log_run_to_mlflow(exp, params=params, metrics=metrics, model=pipe)

def run_cart_regression(X_train, y_train, X_test, y_test, pre):
    exp = "CARTRegression"

    max_depth_list = [None, 5]
    min_samples_split_list = [2, 5]
    min_samples_leaf_list = [1, 2, 5]
    max_features_list = [None, "sqrt", "log2"]  # nº features consideradas al dividir

    for md in max_depth_list:
        for mss in min_samples_split_list:
            for msl in min_samples_leaf_list:
                for mf in max_features_list:
                    params = {
                        "max_depth": md,
                        "min_samples_split": mss,
                        "min_samples_leaf": msl,
                        "max_features": mf,
                        "random_state": 42,
                    }

                    est = DecisionTreeRegressor(**params)
                    pipe = Pipeline([("prep", pre), ("est", est)])

                    # Entrena
                    pipe.fit(X_train, y_train)

                    # Predice
                    y_pred = pipe.predict(X_test)

                    # Métricas
                    metrics = calculate_metrics(y_test, y_pred)
                    print(f"[CART] {params} -> RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f}")
                    log_run_to_mlflow(exp, params=params, metrics=metrics, model=None)

def run_cart_regression2(X_train, y_train, X_test, y_test, pre, cart_params):
    exp = "CARTRegression"

    # --- 1) candidates de ccp_alpha ---
    alphas = cart_params.get("ccp_alpha", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # --- 2) probar cada alpha ---
    for alpha in alphas:
        est = DecisionTreeRegressor(ccp_alpha=float(alpha), random_state=42)
        pipe = Pipeline([("prep", pre), ("est", est)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        params  = {"ccp_alpha": float(alpha), "random_state": 42}

        # Reutiliza tu helper de MLflow
        log_run_to_mlflow(experiment_name=exp,
                          params=params, metrics=metrics, model=None)

        print(f"[CART] ccp_alpha={alpha:.6g} -> RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f} CV={metrics['CV']:.3f}")

def run_random_forest_regression(X_train, y_train, X_test, y_test, pre):
    exp = "RandomForestRegression"

    params = {
        "n_estimators": 100,           # Número de árboles
        "criterion": "squared_error",  
        "max_depth": 10,                # Profundidad máxima de los árboles
        "min_samples_split": 2,        # Mínimo de muestras para dividir un nodo
        "random_state": 42,
        "n_jobs": -1                   # Usa todos los núcleos
    }

    est = RandomForestRegressor(**params)
    pipe = Pipeline([("prep", pre), ("est", est)])

    # Entrena el modelo
    pipe.fit(X_train, y_train)

    # Predicciones
    y_pred = pipe.predict(X_test)

    # Calcula métricas
    metrics = calculate_metrics(y_test, y_pred)

    # Muestra resultados por consola
    print(f"[RandomForest] RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, R2={metrics['R2']:.3f}")

    # Registra en MLflow
    log_run_to_mlflow(exp, params=params, metrics=metrics, model=pipe) 

def run_cubist_regression(X_train, y_train, X_test, y_test, pre, cubist_params):
    exp = "CubistRegression"

    n_committees_list = cubist_params.get("n_committees", [1, 5, 10, 20])
    n_rules_list = cubist_params.get("n_rules", [10, 50, 100])

    for n_committees in n_committees_list:
        for n_rules in n_rules_list:
            params = {
                "n_committees": n_committees,
                "n_rules": n_rules,
            }

            est = Cubist(**params)
            pipe = Pipeline([("prep", pre), ("est", est)])

            # Entrena
            pipe.fit(X_train, y_train)

            # Predice
            y_pred = pipe.predict(X_test)

            # Métricas
            metrics = calculate_metrics(y_test, y_pred)
            print(f"[Cubist] {params} -> RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f}")
            log_run_to_mlflow(exp, params=params, metrics=metrics, model=None)


def main():

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    params = load_params()
    data_path   = params["preprocess"]["output"]
    split_cfg   = params.get("split", {})
    train_cfg = params.get("train", {})
    #method      = split_cfg.get("method", "time")      # 'time' o 'random'
    test_size   = split_cfg.get("test_size", 0.2)
    model_to_run = train_cfg.get("model_to_run", "Cubist")

    # 1 Loads the data
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 2 Splits the data
    X_train, X_test, y_train, y_test = split_train_test_data(df=df, test_size=test_size)

    pre = build_preprocessor()

    if model_to_run == "LinearRegression":
        run_linear_regression(X_train, y_train, X_test, y_test, pre)
    elif model_to_run == "KNN":
        knn_params = params.get("knn", {})
        run_knn_regression(X_train, y_train, X_test, y_test, pre, knn_params)
    elif model_to_run == "CART":
        cart_params = params.get("cart", {})
        run_cart_regression2(X_train, y_train, X_test, y_test, pre, cart_params)
    elif model_to_run == "Cubist":
        cubist_params = params.get("cubist", {})
        run_cubist_regression(X_train, y_train, X_test, y_test, pre, cubist_params)
    elif model_to_run == "RandomForest":
        run_random_forest_regression(X_train, y_train, X_test, y_test, pre)



if __name__ == "__main__":
    main()
