import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

TARGET = "Usage_kWh"

def load_params(path="params.yaml"):
    if Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # defaults si no hay params.yaml
    return {
        "preprocess": {"output": "data/processed/steel_energy_modified_clean.csv"},
        "split": {"method": "time", "test_size": 0.2, "random_state": 42},
        "model": {"n_estimators": 400, "random_state": 42, "log1p_target": True}
    }

def main():
    params = load_params()
    data_path   = params["preprocess"]["output"]
    split_cfg   = params.get("split", {})
    method      = split_cfg.get("method", "time")      # 'time' o 'random'
    test_size   = split_cfg.get("test_size", 0.2)
    random_state= split_cfg.get("random_state", 42)

    # 1) Loads the data
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 2) Separates X/y
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    # 3) Split with scikit-learn
    if method == "time":
        # order by date and NO shuffle (temporal)
        order = df["date"].argsort()
        X = X.iloc[order]
        y = y.iloc[order]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
    elif method == "random":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
    else:
        raise ValueError("split.method debe ser 'time' o 'random'")

    print(f"[split] método={method}  test_size={test_size}  rs={random_state}")
    print(f"[split] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[split] y_train: {y_train.shape} | y_test: {y_test.shape}")

    
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

    mcfg = params.get("model", {})
    rf = RandomForestRegressor(
        n_estimators=mcfg.get("n_estimators", 400),
        random_state=mcfg.get("random_state", 42),
        n_jobs=-1
    )
    if mcfg.get("log1p_target", True):
        est = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    else:
        est = rf

    model = Pipeline([('prep', pre), ('est', est)])

    # 4) Fit the model
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred))
    

    Path("models").mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, "models/rf_model.pkl")


if __name__ == "__main__":
    main()
