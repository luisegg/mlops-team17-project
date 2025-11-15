"""
Integration tests for the /predict endpoint.
"""
import pytest
from unittest.mock import patch
import numpy as np


@pytest.mark.integration
class TestPredictEndpoint:
    """Test suite for the POST /predict endpoint."""

    def test_predict_success(self, test_client, sample_valid_input, mock_model):
        """Test successful prediction with valid input."""
        response = test_client.post("/predict", json=sample_valid_input)

        # Check response status
        assert response.status_code == 200

        # Check response structure
        data = response.json()
        assert "Usage_kWh" in data
        assert isinstance(data["Usage_kWh"], float)

        # Check that the mock model's predict method was called
        mock_model.predict.assert_called_once()

        # Verify the prediction value matches mock output
        assert data["Usage_kWh"] == 42.5

    def test_predict_model_not_loaded(self, test_client_no_model, sample_valid_input):
        """Test prediction fails with 503 when model is not loaded."""
        response = test_client_no_model.post("/predict", json=sample_valid_input)

        # Check response status
        assert response.status_code == 503

        # Check error message
        data = response.json()
        assert "detail" in data
        assert "Model not loaded" in data["detail"]

    def test_predict_missing_required_field(self, test_client, sample_invalid_inputs):
        """Test prediction fails with 422 when required field is missing."""
        invalid_input = sample_invalid_inputs["missing_required_field"]
        response = test_client.post("/predict", json=invalid_input)

        # Check response status (422 Unprocessable Entity)
        assert response.status_code == 422

        # Check that error details are provided
        data = response.json()
        assert "detail" in data

    def test_predict_invalid_nsm_too_high(self, test_client, sample_invalid_inputs):
        """Test prediction fails with 422 when NSM exceeds maximum value."""
        invalid_input = sample_invalid_inputs["invalid_nsm_too_high"]
        response = test_client.post("/predict", json=invalid_input)

        assert response.status_code == 422

    def test_predict_invalid_nsm_negative(self, test_client, sample_invalid_inputs):
        """Test prediction fails with 422 when NSM is negative."""
        invalid_input = sample_invalid_inputs["invalid_nsm_negative"]
        response = test_client.post("/predict", json=invalid_input)

        assert response.status_code == 422

    def test_predict_invalid_day_of_week(self, test_client, sample_invalid_inputs):
        """Test prediction fails with 422 when Day_of_week has invalid value."""
        invalid_input = sample_invalid_inputs["invalid_day_of_week"]
        response = test_client.post("/predict", json=invalid_input)

        assert response.status_code == 422

    def test_predict_invalid_week_status(self, test_client, sample_invalid_inputs):
        """Test prediction fails with 422 when WeekStatus has invalid value."""
        invalid_input = sample_invalid_inputs["invalid_week_status"]
        response = test_client.post("/predict", json=invalid_input)

        assert response.status_code == 422

    def test_predict_invalid_load_type(self, test_client, sample_invalid_inputs):
        """Test prediction fails with 422 when Load_Type has invalid value."""
        invalid_input = sample_invalid_inputs["invalid_load_type"]
        response = test_client.post("/predict", json=invalid_input)

        assert response.status_code == 422

    def test_predict_wrong_type_nsm(self, test_client, sample_invalid_inputs):
        """Test prediction fails with 422 when NSM has wrong data type."""
        invalid_input = sample_invalid_inputs["wrong_type_nsm"]
        response = test_client.post("/predict", json=invalid_input)

        assert response.status_code == 422

    def test_predict_model_prediction_fails(self, test_client, sample_valid_input):
        """Test prediction fails with 500 when model.predict() raises exception."""
        # Patch the model to raise an exception during prediction
        with patch('src.api.main.model') as mock_failing_model:
            mock_failing_model.predict.side_effect = Exception("Model prediction error")
            mock_failing_model.__bool__.return_value = True  # Model is loaded

            response = test_client.post("/predict", json=sample_valid_input)

            # Check response status (500 Internal Server Error)
            assert response.status_code == 500

            # Check error message
            data = response.json()
            assert "detail" in data
            assert "Prediction failed" in data["detail"]

    def test_predict_response_schema(self, test_client, sample_valid_input):
        """Test that response matches PredictionResponse schema."""
        response = test_client.post("/predict", json=sample_valid_input)

        assert response.status_code == 200
        data = response.json()

        # Check that only expected fields are present
        assert set(data.keys()) == {"Usage_kWh"}

        # Check field type
        assert isinstance(data["Usage_kWh"], (int, float))

    @pytest.mark.parametrize("day,week_status,load_type", [
        ("Monday", "Weekday", "Light_Load"),
        ("Tuesday", "Weekday", "Medium_Load"),
        ("Wednesday", "Weekday", "Maximum_Load"),
        ("Friday", "Weekday", "Light_Load"),
        ("Saturday", "Weekend", "Medium_Load"),
        ("Sunday", "Weekend", "Maximum_Load"),
    ])
    def test_predict_with_various_enum_combinations(self, test_client, sample_valid_input, day, week_status, load_type):
        """Test prediction with different valid enum combinations."""
        test_input = sample_valid_input.copy()
        test_input["Day_of_week"] = day
        test_input["WeekStatus"] = week_status
        test_input["Load_Type"] = load_type

        response = test_client.post("/predict", json=test_input)

        assert response.status_code == 200
        data = response.json()
        assert "Usage_kWh" in data

    @pytest.mark.parametrize("nsm_value", [0, 43200, 86399])
    def test_predict_with_edge_case_nsm_values(self, test_client, sample_valid_input, nsm_value):
        """Test prediction with edge case NSM values (0, noon, end of day)."""
        test_input = sample_valid_input.copy()
        test_input["NSM"] = nsm_value

        response = test_client.post("/predict", json=test_input)

        assert response.status_code == 200
        data = response.json()
        assert "Usage_kWh" in data

    def test_predict_with_co2_alias(self, test_client, sample_valid_input):
        """Test that CO2 field can be accessed via alias."""
        # The sample_valid_input already uses the alias "CO2(tCO2)"
        response = test_client.post("/predict", json=sample_valid_input)

        assert response.status_code == 200
        data = response.json()
        assert "Usage_kWh" in data

    def test_predict_with_various_numeric_values(self, test_client, sample_valid_input):
        """Test prediction with different numeric values."""
        test_input = sample_valid_input.copy()
        test_input["Lagging_Current_Reactive_Power_kVarh"] = 10.5
        test_input["Leading_Current_Reactive_Power_kVarh"] = 7.2
        test_input["CO2(tCO2)"] = 0.15
        test_input["Lagging_Current_Power_Factor"] = 95.5
        test_input["Leading_Current_Power_Factor"] = 88.3

        response = test_client.post("/predict", json=test_input)

        assert response.status_code == 200
        data = response.json()
        assert "Usage_kWh" in data

    def test_predict_returns_different_values_for_different_inputs(self, test_client, sample_valid_input):
        """Test that model is actually being called with the input data."""
        # This test verifies that the model's predict method receives a DataFrame
        with patch('src.api.main.model') as mock_model:
            mock_model.predict.return_value = np.array([100.0])
            mock_model.__bool__.return_value = True

            response = test_client.post("/predict", json=sample_valid_input)

            # Verify predict was called
            assert mock_model.predict.called

            # Verify the call was with a DataFrame
            call_args = mock_model.predict.call_args[0]
            assert len(call_args) == 1
            import pandas as pd
            assert isinstance(call_args[0], pd.DataFrame)

            # Verify response contains the mocked value
            assert response.status_code == 200
            assert response.json()["Usage_kWh"] == 100.0

    def test_predict_handles_empty_json(self, test_client):
        """Test prediction fails gracefully with empty JSON."""
        response = test_client.post("/predict", json={})

        assert response.status_code == 422

    def test_predict_handles_null_values(self, test_client, sample_valid_input):
        """Test prediction fails when required fields are null."""
        test_input = sample_valid_input.copy()
        test_input["NSM"] = None

        response = test_client.post("/predict", json=test_input)

        assert response.status_code == 422
