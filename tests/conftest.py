"""
Shared pytest fixtures for API testing.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import numpy as np


@pytest.fixture
def mock_model():
    """
    Create a mock ML model that mimics the Cubist model interface.

    Returns:
        Mock object with predict method that returns a test prediction.
    """
    mock = Mock()
    mock.predict.return_value = np.array([42.5])  # Mock prediction value
    return mock


@pytest.fixture
def sample_valid_input():
    """
    Provide a valid prediction input for testing.

    Returns:
        Dictionary with valid input data matching PredictionInput schema.
    """
    return {
        "Lagging_Current_Reactive_Power_kVarh": 5.0,
        "Leading_Current_Reactive_Power_kVarh": 3.5,
        "CO2(tCO2)": 0.05,
        "Lagging_Current_Power_Factor": 85.2,
        "Leading_Current_Power_Factor": 92.1,
        "NSM": 43200,  # Noon (12:00 PM)
        "Day_of_week": "Monday",
        "WeekStatus": "Weekday",
        "Load_Type": "Medium_Load"
    }


@pytest.fixture
def sample_invalid_inputs():
    """
    Provide various invalid inputs for error testing.

    Returns:
        Dictionary of test case names to invalid input data.
    """
    return {
        "missing_required_field": {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            # Missing other required fields
        },
        "invalid_nsm_too_high": {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            "Leading_Current_Reactive_Power_kVarh": 3.5,
            "CO2(tCO2)": 0.05,
            "Lagging_Current_Power_Factor": 85.2,
            "Leading_Current_Power_Factor": 92.1,
            "NSM": 90000,  # Invalid: exceeds 86399
            "Day_of_week": "Monday",
            "WeekStatus": "Weekday",
            "Load_Type": "Medium_Load"
        },
        "invalid_nsm_negative": {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            "Leading_Current_Reactive_Power_kVarh": 3.5,
            "CO2(tCO2)": 0.05,
            "Lagging_Current_Power_Factor": 85.2,
            "Leading_Current_Power_Factor": 92.1,
            "NSM": -100,  # Invalid: negative value
            "Day_of_week": "Monday",
            "WeekStatus": "Weekday",
            "Load_Type": "Medium_Load"
        },
        "invalid_day_of_week": {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            "Leading_Current_Reactive_Power_kVarh": 3.5,
            "CO2(tCO2)": 0.05,
            "Lagging_Current_Power_Factor": 85.2,
            "Leading_Current_Power_Factor": 92.1,
            "NSM": 43200,
            "Day_of_week": "InvalidDay",  # Invalid enum value
            "WeekStatus": "Weekday",
            "Load_Type": "Medium_Load"
        },
        "invalid_week_status": {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            "Leading_Current_Reactive_Power_kVarh": 3.5,
            "CO2(tCO2)": 0.05,
            "Lagging_Current_Power_Factor": 85.2,
            "Leading_Current_Power_Factor": 92.1,
            "NSM": 43200,
            "Day_of_week": "Monday",
            "WeekStatus": "NotAWeekStatus",  # Invalid enum value
            "Load_Type": "Medium_Load"
        },
        "invalid_load_type": {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            "Leading_Current_Reactive_Power_kVarh": 3.5,
            "CO2(tCO2)": 0.05,
            "Lagging_Current_Power_Factor": 85.2,
            "Leading_Current_Power_Factor": 92.1,
            "NSM": 43200,
            "Day_of_week": "Monday",
            "WeekStatus": "Weekday",
            "Load_Type": "Invalid_Load"  # Invalid enum value
        },
        "wrong_type_nsm": {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            "Leading_Current_Reactive_Power_kVarh": 3.5,
            "CO2(tCO2)": 0.05,
            "Lagging_Current_Power_Factor": 85.2,
            "Leading_Current_Power_Factor": 92.1,
            "NSM": "not_an_integer",  # Invalid: string instead of int
            "Day_of_week": "Monday",
            "WeekStatus": "Weekday",
            "Load_Type": "Medium_Load"
        }
    }


@pytest.fixture
def test_client(mock_model):
    """
    Create a FastAPI TestClient with a mocked model.

    This fixture patches the global model in main.py and provides a test client
    for making requests to the API.

    Args:
        mock_model: The mocked ML model fixture

    Returns:
        TestClient instance with mocked dependencies
    """
    # Patch both the model and model_metadata at module level
    with patch('src.api.main.model', mock_model), \
         patch('src.api.main.model_metadata', {'model_name': 'test_model'}):
        from src.api.main import app
        client = TestClient(app)
        yield client


@pytest.fixture
def test_client_no_model():
    """
    Create a FastAPI TestClient with no model loaded (model = None).

    This simulates the scenario where model loading failed.

    Returns:
        TestClient instance with model set to None
    """
    with patch('src.api.main.model', None), \
         patch('src.api.main.model_metadata', {}):
        from src.api.main import app
        client = TestClient(app)
        yield client
