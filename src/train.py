import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

TARGET = "Usage_kWh"

def load_params(path="params.yaml"):
    if Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # defaults si no hay params.yaml
    return {
        "preprocess": {"output": "data/processed/steel_energy_modified_clean.csv"},
        "split": {"method": "time", "test_size": 0.2, "random_state": 42}
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

    # Here, in the next step, you will add:
    # - your ColumnTransformer
    # - the model (fit in train, eval in test)
    # - save the model and metrics

if __name__ == "__main__":
    main()
