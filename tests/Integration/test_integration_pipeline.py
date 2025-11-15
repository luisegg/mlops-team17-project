import pytest
from src.train import FeaturePreprocessor, DataSplitter, ModelTrainer, HyperparameterManager
from tests.conftest import input_df
from src.train import TARGET

@pytest.mark.integration
def test_pipeline_end_to_end(input_df):
    """
    Test de integración extremo a extremo:
    carga de datos → limpieza → preprocesamiento → split → entrenamiento → métricas.
    """
    # Usar copia del DataFrame para no modificar fixture
    df = input_df.copy()

    # 🔧 Patch columnas para compatibilidad con FeaturePreprocessor
    df.rename(columns={
        "Lagging_Current_Reactive_Power_kVarh": "Lagging_Current_Reactive.Power_kVarh"
    }, inplace=True)
    
    # 1️⃣ Inicializar el preprocesador real
    preprocessor = FeaturePreprocessor().build_preprocessor()

    # 2️⃣ Dividir los datos
    splitter = DataSplitter(target_column=TARGET)
    X_train, X_test, y_train, y_test = splitter.split_data(df, test_size=0.2)

    # 3️⃣ Obtener configuración de modelo (Cubist como ejemplo)
    hyper_manager = HyperparameterManager(params_path="params.yaml")
    cubist_config = hyper_manager.get_model_config("Cubist")

    # 4️⃣ Entrenar el modelo usando ModelTrainer
    trainer = ModelTrainer()
    trained_models = trainer.train_model(
        model_config=cubist_config,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor
    )

    # 5️⃣ Validaciones básicas
    # Verificar que se entrenaron modelos
    assert len(trained_models) > 0, "No se entrenó ningún modelo"

    # Verificar que se generaron métricas para cada modelo entrenado
    for model_dict in trained_models:
        assert "model" in model_dict, "Falta la clave 'model' en el resultado"
        assert "metrics" in model_dict, "Falta la clave 'metrics' en el resultado"
        metrics = model_dict["metrics"]
        required_keys = ["RMSE", "MAE", "R2", "CV"]
        missing_keys = [key for key in required_keys if key not in metrics]
        assert not missing_keys, f"Faltan métricas en el resultado: {missing_keys}"
