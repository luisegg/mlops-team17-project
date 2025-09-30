from errno import ESTALE
import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold
from sklearn.linear_model import LinearRegression
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

def build_rf_model(params):

    rf = RandomForestRegressor(
        n_estimators=params.get("model", {}).get("n_estimators", 400),
        random_state=params.get("model", {}).get("random_state", 42),
        n_jobs=-1
    )
    # Opción: log1p al target
    if params.get("model", {}).get("log1p_target", True):
        return TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    return rf

def build_cv(cfg):
    cv_cfg = cfg.get("cv", {}) if cfg else {}
    kind = str(cv_cfg.get("kind", "timeseries")).lower()
    n_splits = int(cv_cfg.get("n_splits", 5))
    if kind == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    # kfold por si quieres aleatorio
    shuffle = bool(cv_cfg.get("shuffle", True))
    rs = cv_cfg.get("random_state", 42)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs)

def build_glr_model():
    return LinearRegression(fit_intercept=False)


def main():
    params = load_params()
    data_path   = params["preprocess"]["output"]
    split_cfg   = params.get("split", {})
    #method      = split_cfg.get("method", "time")      # 'time' o 'random'
    test_size   = split_cfg.get("test_size", 0.2)
    random_state= split_cfg.get("random_state", 42)

    # 1 Loads the data
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 2 Splits the data
    X_train, X_test, y_train, y_test = split_train_test_data(df=df, test_size=test_size)

    pre = build_preprocessor()

    # Configure MLflow tracking URI from env (already set above)
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    models = ['glr']
    for model in models:
        with mlflow.start_run():            
            mlflow.set_experiment('Linear Regression')

            #est = build_rf_model(params)
            # Builds the model
            est = build_glr_model()

            # Creates a pipeline
            pipe = Pipeline([('prep', pre), ('est', est)])

            # Fit the model
            pipe.fit(X_train, y_train)

            #Predicts
            y_pred = pipe.predict(X_test)

            '''
            signature = infer_signature(X_train, y_train)

            model_info = mlflow.sklearn.log_model(
                sk_model = est,
                artifact_path='glr_model',
                signature = signature,
                input_example = X_train,
                #registered_model_name = 'Linear Regression'
            )
            '''

            # Calculates a logs metrics
            RMSE = mean_squared_error(y_test, y_pred)
            MAE  = mean_absolute_error(y_test, y_pred)
            R2   = r2_score(y_test, y_pred)
            mlflow.log_metric('RMSE', RMSE)
            mlflow.log_metric('MAE', MAE)
            mlflow.log_metric('R2', R2)
            print('RMSE:', RMSE)
            print('MAE:', MAE)
            print('R2:', R2)

    '''
    Path("models").mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, "models/rf_model.pkl")
    '''

if __name__ == "__main__":
    main()
