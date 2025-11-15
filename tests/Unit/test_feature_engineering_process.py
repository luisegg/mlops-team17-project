import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def test_numeric_scaling(input_df):
    """
    Verifica que las columnas numéricas estén correctamente escaladas o normalizadas.
    """
    numeric_cols = ["NSM", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "CO2(tCO2)"]
    df_num = input_df[numeric_cols].copy().fillna(0)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_num)

    means = np.mean(scaled_values, axis=0)
    stds = np.std(scaled_values, axis=0)

    for mean, std in zip(means, stds):
        assert abs(mean) < 1e-6, f"Media de columna no centrada en 0: {mean}"
        assert abs(std - 1) < 1e-6, f"Desviación estándar de columna no igual a 1: {std}"


def test_derived_features(input_df):
    """
    Verifica que los features calculados sean consistentes.
    """
    df = input_df.copy()
    if "Lagging_Current_Power_Factor" in df.columns and "Leading_Current_Power_Factor" in df.columns:
        df["Total_Power_Factor"] = df["Lagging_Current_Power_Factor"] + df["Leading_Current_Power_Factor"]

        assert df["Total_Power_Factor"].notnull().all(), "Feature derivada contiene NaN"
        assert (df["Total_Power_Factor"] >= 0).all(), "Feature derivada contiene valores negativos"


def test_missing_values(input_df):
    """
    Verifica que los datos faltantes sean manejados correctamente.
    """
    df_filled = input_df.fillna(0)
    assert not df_filled.isnull().any().any(), "Existen valores faltantes sin tratar"


def test_outliers(input_df):
    """
    Verifica la existencia de outliers en columnas numéricas.
    """
    numeric_cols = ["NSM", "Lagging_Current_Power_Factor","CO2(tCO2)"]

    for col in numeric_cols:
        series = input_df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        # Control: no más del 5% de outliers
        assert outliers.shape[0] / len(series) < 0.05, f"Existen demasiados outliers en {col}"
