"""
Unit tests for API helper functions.
"""
import pytest
import pandas as pd
from src.api.main import input_to_dataframe
from src.api.schemas import PredictionInput


@pytest.mark.unit
class TestInputToDataFrame:
    """Test suite for the input_to_dataframe function."""

    def test_input_to_dataframe_basic(self, sample_valid_input):
        """Test that input_to_dataframe creates a DataFrame with correct structure."""
        # Create PredictionInput from sample data
        prediction_input = PredictionInput(**sample_valid_input)

        # Convert to DataFrame
        df = input_to_dataframe(prediction_input)

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Should have exactly one row

    def test_input_to_dataframe_column_names(self, sample_valid_input):
        """Test that DataFrame has the correct column names expected by the model."""
        prediction_input = PredictionInput(**sample_valid_input)
        df = input_to_dataframe(prediction_input)

        expected_columns = [
            'date',
            'Lagging_Current_Reactive.Power_kVarh',  # Note the dot instead of underscore
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor',
            'NSM',
            'Day_of_week',
            'WeekStatus',
            'Load_Type'
        ]

        assert list(df.columns) == expected_columns

    def test_input_to_dataframe_data_types(self, sample_valid_input):
        """Test that DataFrame columns have correct data types."""
        prediction_input = PredictionInput(**sample_valid_input)
        df = input_to_dataframe(prediction_input)

        # Check numeric types
        assert df['Lagging_Current_Reactive.Power_kVarh'].dtype in [float, 'float64']
        assert df['Leading_Current_Reactive_Power_kVarh'].dtype in [float, 'float64']
        assert df['CO2(tCO2)'].dtype in [float, 'float64']
        assert df['NSM'].dtype in [int, 'int64']

        # Check that enum values are strings
        assert isinstance(df['Day_of_week'].iloc[0], str)
        assert isinstance(df['WeekStatus'].iloc[0], str)
        assert isinstance(df['Load_Type'].iloc[0], str)

    def test_input_to_dataframe_values(self, sample_valid_input):
        """Test that DataFrame contains the correct values from input."""
        prediction_input = PredictionInput(**sample_valid_input)
        df = input_to_dataframe(prediction_input)

        # Verify values match input
        assert df['Lagging_Current_Reactive.Power_kVarh'].iloc[0] == sample_valid_input['Lagging_Current_Reactive_Power_kVarh']
        assert df['Leading_Current_Reactive_Power_kVarh'].iloc[0] == sample_valid_input['Leading_Current_Reactive_Power_kVarh']
        assert df['CO2(tCO2)'].iloc[0] == sample_valid_input['CO2(tCO2)']
        assert df['Lagging_Current_Power_Factor'].iloc[0] == sample_valid_input['Lagging_Current_Power_Factor']
        assert df['Leading_Current_Power_Factor'].iloc[0] == sample_valid_input['Leading_Current_Power_Factor']
        assert df['NSM'].iloc[0] == sample_valid_input['NSM']
        assert df['Day_of_week'].iloc[0] == sample_valid_input['Day_of_week']
        assert df['WeekStatus'].iloc[0] == sample_valid_input['WeekStatus']
        assert df['Load_Type'].iloc[0] == sample_valid_input['Load_Type']

    def test_input_to_dataframe_has_dummy_date(self, sample_valid_input):
        """Test that DataFrame includes a dummy date column."""
        prediction_input = PredictionInput(**sample_valid_input)
        df = input_to_dataframe(prediction_input)

        # Check that date column exists and is a Timestamp
        assert 'date' in df.columns
        assert isinstance(df['date'].iloc[0], pd.Timestamp)

    @pytest.mark.parametrize("day,week_status,load_type", [
        ("Monday", "Weekday", "Light_Load"),
        ("Friday", "Weekday", "Medium_Load"),
        ("Saturday", "Weekend", "Maximum_Load"),
        ("Sunday", "Weekend", "Light_Load"),
    ])
    def test_input_to_dataframe_with_different_enums(self, sample_valid_input, day, week_status, load_type):
        """Test that different enum combinations are handled correctly."""
        # Modify sample input with different enum values
        test_input = sample_valid_input.copy()
        test_input['Day_of_week'] = day
        test_input['WeekStatus'] = week_status
        test_input['Load_Type'] = load_type

        prediction_input = PredictionInput(**test_input)
        df = input_to_dataframe(prediction_input)

        # Verify enum values are preserved
        assert df['Day_of_week'].iloc[0] == day
        assert df['WeekStatus'].iloc[0] == week_status
        assert df['Load_Type'].iloc[0] == load_type

    def test_input_to_dataframe_with_edge_case_nsm(self, sample_valid_input):
        """Test that edge case NSM values (0 and 86399) are handled correctly."""
        # Test NSM = 0 (midnight)
        test_input_midnight = sample_valid_input.copy()
        test_input_midnight['NSM'] = 0
        prediction_input = PredictionInput(**test_input_midnight)
        df = input_to_dataframe(prediction_input)
        assert df['NSM'].iloc[0] == 0

        # Test NSM = 86399 (23:59:59)
        test_input_end_of_day = sample_valid_input.copy()
        test_input_end_of_day['NSM'] = 86399
        prediction_input = PredictionInput(**test_input_end_of_day)
        df = input_to_dataframe(prediction_input)
        assert df['NSM'].iloc[0] == 86399

    def test_input_to_dataframe_co2_alias_handling(self, sample_valid_input):
        """Test that CO2 field alias is handled correctly."""
        prediction_input = PredictionInput(**sample_valid_input)
        df = input_to_dataframe(prediction_input)

        # The DataFrame column should use the alias format: CO2(tCO2)
        assert 'CO2(tCO2)' in df.columns
        assert df['CO2(tCO2)'].iloc[0] == sample_valid_input['CO2(tCO2)']
