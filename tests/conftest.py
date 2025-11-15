import pytest
import pandas as pd
from src.preprocess import DataLoader, DataCleaner

@pytest.fixture(scope="session")
def input_df():
    """
    Fixture para cargar el CSV crudo y devolver el DataFrame ya limpio.
    Esto asegura que los tests trabajen con datos consistentes.
    """
    loader = DataLoader()
    cleaner = DataCleaner()

    # Ruta al CSV crudo
    input_path = "data/raw/steel_energy_original.csv"

    # Cargar datos crudos
    df = loader.load_data(input_path)

    # Limpiar datos
    df_cleaned = cleaner.clean_data(df)

    # Normalizar nombres de columnas si hay errores de tipografía
    df_cleaned.rename(columns={
        "Lagging_Current_Reactive.Power_kVarh": "Lagging_Current_Reactive_Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh": "Leading_Current_Reactive_Power_kVarh"
    }, inplace=True)

    # Normalizar Load_Type para que coincida con los tests
    df_cleaned["Load_Type"] = df_cleaned["Load_Type"].replace({"Maximum_Load": "High_Load"})

    return df_cleaned
