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
    rmse = root_mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('R2:', r2)

    metrics = {"RMSE" : rmse, "MAE": mae, "R2": r2}

    log_run_to_mlflow(exp, params={}, metrics=metrics, model=pipe)

def run_knn_regression(X_train, y_train, X_test, y_test, pre):
    exp = "KNNRegression"
   
    k_list = [3, 5, 7, 11]
    weights_list = ["uniform", "distance"]
    p_list = [1, 2]            # 1=Manhattan, 2=Euclidiana
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
                rmse = root_mean_squared_error(y_test, y_pred)
                mae  = mean_absolute_error(y_test, y_pred)
                r2   = r2_score(y_test, y_pred)
                print('RMSE:', rmse)
                print('MAE:', mae)
                print('R2:', r2)

                metrics = {"RMSE" : rmse, "MAE": mae, "R2": r2}

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

                    # Métricas (RMSE real: squared=False)
                    rmse = root_mean_squared_error(y_test, y_pred)
                    mae  = mean_absolute_error(y_test, y_pred)
                    r2   = r2_score(y_test, y_pred)
                    print(f"[CART] {params} -> RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")

                    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
                    log_run_to_mlflow(exp, params=params, metrics=metrics, model=None)

def run_cart_regression2(X_tr, y_tr, X_te, y_te, pre):
    exp = "CARTRegression"

    # --- 1) candidates de ccp_alpha ---
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # --- 2) probar cada alpha ---
    for alpha in alphas:
        est = DecisionTreeRegressor(ccp_alpha=float(alpha), random_state=42)
        pipe = Pipeline([("prep", pre), ("est", est)])

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        # RMSE compatible con cualquier versión de sklearn
        rmse = float(root_mean_squared_error(y_te, y_pred))
        mae  = float(mean_absolute_error(y_te, y_pred))
        r2   = float(r2_score(y_te, y_pred))

        params  = {"ccp_alpha": float(alpha), "random_state": 42}
        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

        # Reutiliza tu helper de MLflow
        log_run_to_mlflow(experiment_name=exp,
                          params=params, metrics=metrics, model=None)

        print(f"[CART] ccp_alpha={alpha:.6g} -> RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")


def main():

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    params = load_params()
    data_path   = params["preprocess"]["output"]
    split_cfg   = params.get("split", {})
    #method      = split_cfg.get("method", "time")      # 'time' o 'random'
    test_size   = split_cfg.get("test_size", 0.2)

    # 1 Loads the data
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 2 Splits the data
    X_train, X_test, y_train, y_test = split_train_test_data(df=df, test_size=test_size)

    pre = build_preprocessor()

    #run_linear_regression(X_train, y_train, X_test, y_test, pre)
    #run_knn_regression(X_train, y_train, X_test, y_test, pre)
    #run_cart_regression(X_train, y_train, X_test, y_test, pre)
    run_cart_regression2(X_train, y_train, X_test, y_test, pre)



if __name__ == "__main__":
    main()
