import pytest
import pandas as pd

# ---------------------------------------------------------
# 1. VALIDACIÓN DE COLUMNAS
# ---------------------------------------------------------
def test_columns_exist(input_df):
    """Verifica que todas las columnas esperadas estén en el dataset."""
    expected_cols = {
        "date",
        "Usage_kWh",
        "Lagging_Current_Reactive_Power_kVarh",   # ✅ columna con guion bajo
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
        "WeekStatus",
        "Day_of_week",
        "Load_Type",
    }

    missing = expected_cols - set(input_df.columns)
    assert not missing, f"Missing required column(s): {missing}"


# ---------------------------------------------------------
# 2. VALIDACIÓN DE TIPOS NUMÉRICOS
# ---------------------------------------------------------
def test_column_types(input_df):
    """Convierte columnas numéricas y valida que no contengan strings."""
    numeric_cols = [
        "Usage_kWh",
        "Lagging_Current_Reactive_Power_kVarh",  # ✅ columna con guion bajo
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
    ]

    df = input_df.copy()

    # Convertir a numérico
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Si hay NaN → había strings o valores inválidos
    assert df[numeric_cols].notnull().all().all(), \
        f"Columns contain non-numeric values: {df[numeric_cols].isnull().sum().to_dict()}"


# ---------------------------------------------------------
# 3. VALIDACIÓN DE AUSENCIA DE NULOS
# ---------------------------------------------------------
def test_no_missing_values(input_df):
    """Verifica que el dataset no tenga valores nulos."""
    total_missing = input_df.isnull().sum().sum()
    assert total_missing == 0, f"Hay valores nulos en el dataset: {input_df.isnull().sum()}"


# ---------------------------------------------------------
# 4. VALIDACIÓN DEL RANGO DE NSM
# ---------------------------------------------------------
def test_nsm_range(input_df):
    """NSM debe estar entre 0 y 86400 (segundos del día)."""
    df = input_df.copy()
    df["NSM"] = pd.to_numeric(df["NSM"], errors="coerce")
    assert df["NSM"].between(0, 86400).all(), "NSM contiene valores fuera de rango"


# ---------------------------------------------------------
# 5. FACTORES DE POTENCIA NO NEGATIVOS
# ---------------------------------------------------------
def test_power_factors_non_negative(input_df):
    df = input_df.copy()
    cols = [
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
    ]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        assert (df[col] >= 0).all(), f"{col} contiene valores negativos"


# ---------------------------------------------------------
# 6. CO2 NO NEGATIVO
# ---------------------------------------------------------
def test_co2_positive(input_df):
    df = input_df.copy()
    df["CO2(tCO2)"] = pd.to_numeric(df["CO2(tCO2)"], errors="coerce")
    assert (df["CO2(tCO2)"] >= 0).all(), "CO2 contiene valores negativos"


# ---------------------------------------------------------
# 7. DÍAS DE LA SEMANA VÁLIDOS
# ---------------------------------------------------------
def test_day_of_week_valid(input_df):
    valid_days = {
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    }
    assert set(input_df["Day_of_week"].unique()).issubset(valid_days), \
        "Day_of_week contiene valores no válidos"


# ---------------------------------------------------------
# 8. WEEKSTATUS VÁLIDO
# ---------------------------------------------------------
def test_weekstatus_valid(input_df):
    valid = {"Weekday", "Weekend"}
    assert set(input_df["WeekStatus"].unique()).issubset(valid), \
        "WeekStatus contiene valores no válidos"


# ---------------------------------------------------------
# 9. LOAD TYPES VÁLIDOS
# ---------------------------------------------------------
def test_load_type_valid(input_df):
    valid = {"Light_Load", "Medium_Load", "High_Load"}
    assert set(input_df["Load_Type"].unique()).issubset(valid), \
        "Load_Type contiene valores no válidos"
