import yaml
import pytest

# Mapeo entre nombres esperados por los tests y las claves reales del YAML
MODEL_KEY_MAP = {
    "LinearRegression": "linear_regression",
    "KNN": "knn",
    "CART": "cart",
    "RandomForest": "random_forest",
    "Cubist": "cubist",
    "SVM": "svm"
}

@pytest.fixture
def configs():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def test_model_configs_structure(configs):
    for model, yaml_key in MODEL_KEY_MAP.items():
        assert yaml_key in configs, f"No se encontró la sección '{yaml_key}' en params.yaml"
        assert isinstance(configs[yaml_key], dict), f"{yaml_key} no es un dict"

def test_hyperparameters_not_empty(configs):
    for model, yaml_key in MODEL_KEY_MAP.items():
        params = configs[yaml_key]
        assert len(params) > 0, f"{model} no tiene hiperparámetros"

def test_no_bad_hyperparameters(configs):
    for model, yaml_key in MODEL_KEY_MAP.items():
        params = configs[yaml_key]
        for key, value in params.items():
            assert value is not None, f"{model} tiene hiperparámetro nulo: {key}"

def test_hyperparameter_logical_coherence(configs):
    for model, yaml_key in MODEL_KEY_MAP.items():
        params = configs[yaml_key]

        for key, value in params.items():
            if isinstance(value, list):
                assert len(value) > 0, f"{model}: la lista de {key} está vacía"
