"""
Pydantic schemas for FastAPI Steel Energy Prediction Service
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from enum import Enum


class DayOfWeek(str, Enum):
    """Valid days of the week"""
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"


class WeekStatusEnum(str, Enum):
    """Valid week status values"""
    WEEKDAY = "Weekday"
    WEEKEND = "Weekend"


class LoadTypeEnum(str, Enum):
    """Valid load type categories"""
    LIGHT_LOAD = "Light_Load"
    MEDIUM_LOAD = "Medium_Load"
    MAXIMUM_LOAD = "Maximum_Load"


class PredictionInput(BaseModel):
    """
    Input features for steel energy consumption prediction.

    All features are required and must match the schema used during model training.
    """
    model_config = ConfigDict(populate_by_name=True)

    Lagging_Current_Reactive_Power_kVarh: float = Field(
        description="Lagging current reactive power in kVArh"
    )
    Leading_Current_Reactive_Power_kVarh: float = Field(
        description="Leading current reactive power in kVArh"
    )
    CO2_tCO2: float = Field(
        description="CO2 emissions in tons",
        alias="CO2(tCO2)"
    )
    Lagging_Current_Power_Factor: float = Field(
        description="Lagging current power factor",
        ge=0.0,
        le=1.0
    )
    Leading_Current_Power_Factor: float = Field(
        description="Leading current power factor",
        ge=0.0,
        le=1.0
    )
    NSM: int = Field(
        description="Number of seconds from midnight (0-86399)",
        ge=0,
        le=86399
    )
    Day_of_week: DayOfWeek = Field(
        description="Day of the week"
    )
    WeekStatus: WeekStatusEnum = Field(
        description="Week status (Weekday or Weekend)"
    )
    Load_Type: LoadTypeEnum = Field(
        description="Type of load (Light, Medium, or Maximum)"
    )


class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    Usage_kWh: float = Field(
        description="Predicted energy usage in kilowatt-hours"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    instances: List[PredictionInput] = Field(
        description="List of input instances for prediction",
        min_length=1
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse] = Field(
        description="List of predictions corresponding to input instances"
    )
    count: int = Field(
        description="Number of predictions returned"
    )


class ModelInfo(BaseModel):
    """Model metadata information"""
    model_name: str = Field(description="Name of the model in MLflow registry")
    model_version: Optional[str] = Field(default=None, description="Model version or stage")
    model_uri: str = Field(description="MLflow model URI")
    run_id: Optional[str] = Field(default=None, description="MLflow run ID")
    parameters: Optional[dict] = Field(default=None, description="Model hyperparameters")
    metrics: Optional[dict] = Field(default=None, description="Model performance metrics")
    model_type: str = Field(default="Cubist", description="Type of model algorithm")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service health status")
    model_loaded: bool = Field(description="Whether model is loaded successfully")
    model_name: Optional[str] = Field(default=None, description="Name of loaded model")
