import pytest
import numpy as np
from src.train import MODEL_CONFIGS

@pytest.fixture
def synthetic_data():
    """Genera un dataset sintético pequeño y determinístico."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = X[:, 0] * 3 + np.random.randn(20) * 0.1
    return X, y


@pytest.mark.parametrize("model_name", [
    "LinearRegression",
    "KNN",
    "CART",
    "RandomForest",
    "Cubist",
    "SVM"
])
def test_model_can_train_and_predict(model_name, synthetic_data):
    X, y = synthetic_data

    cfg = MODEL_CONFIGS[model_name]

    # Instanciar modelo
    estimator_cls = cfg["estimator"]
    model = estimator_cls()

    # Entrenar
    model.fit(X, y)

    # Predecir
    preds = model.predict(X)

    # Validaciones de calidad
    assert preds is not None
    assert len(preds) == len(y)
    assert not np.isnan(preds).any(), "El modelo devuelve NaNs"
    assert not np.isinf(preds).any(), "El modelo devuelve Inf"


def test_model_reproducibility(synthetic_data):
    """Validar reproducibilidad para cualquier modelo con random_state."""
    X, y = synthetic_data

    rf_cfg = MODEL_CONFIGS["RandomForest"]
    rf = rf_cfg["estimator"](random_state=42)

    rf.fit(X, y)
    preds1 = rf.predict(X)

    rf2 = rf_cfg["estimator"](random_state=42)
    rf2.fit(X, y)
    preds2 = rf2.predict(X)

    np.testing.assert_array_almost_equal(
        preds1, preds2, decimal=5,
        err_msg="RandomForest no es reproducible aun con random_state"
    )
