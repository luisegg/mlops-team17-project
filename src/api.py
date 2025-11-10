"""
FastAPI service for Steel Energy Consumption Prediction using Cubist Model

This service exposes a REST API to make predictions using a Cubist regression model
trained on steel industry energy consumption data and stored in MLflow Model Registry.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.schemas import (
    PredictionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
model_metadata = {}


def load_model_from_registry(
    model_name: str = "steel-energy-cubist",
    stage: str = "Production"
) -> tuple:
    """
    Load model from MLflow Model Registry.

    Args:
        model_name: Name of the model in MLflow registry
        stage: Model stage (Production, Staging, None) or version number

    Returns:
        Tuple of (model, metadata_dict)

    Raises:
        Exception: If model cannot be loaded
    """
    logger.info(f"Loading model '{model_name}' from MLflow registry (stage: {stage})...")

    # Load environment variables
    load_dotenv()

    # Set MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not found in environment variables")

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Construct model URI
    model_uri = f"models:/{model_name}/{stage}"

    try:
        # Load the model
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded successfully from {model_uri}")

        # Get model metadata
        client = mlflow.tracking.MlflowClient()

        # Get model version details
        try:
            model_versions = client.get_latest_versions(model_name, stages=[stage])
            if model_versions:
                model_version = model_versions[0]
                run_id = model_version.run_id
                version = model_version.version

                # Get run details for parameters and metrics
                run = client.get_run(run_id)

                metadata = {
                    "model_name": model_name,
                    "model_version": str(version),
                    "model_stage": stage,
                    "model_uri": model_uri,
                    "run_id": run_id,
                    "parameters": dict(run.data.params),
                    "metrics": dict(run.data.metrics),
                    "model_type": "Cubist"
                }
            else:
                metadata = {
                    "model_name": model_name,
                    "model_stage": stage,
                    "model_uri": model_uri,
                    "model_type": "Cubist"
                }
        except Exception as e:
            logger.warning(f"Could not retrieve model metadata: {e}")
            metadata = {
                "model_name": model_name,
                "model_uri": model_uri,
                "model_type": "Cubist"
            }

        return loaded_model, metadata

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the model on startup and cleans up on shutdown.
    """
    global model, model_metadata

    # Startup: Load model
    logger.info("Starting up FastAPI service...")
    try:
        model, model_metadata = load_model_from_registry()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("Service will start but predictions will fail until model is loaded")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down FastAPI service...")
    model = None
    model_metadata = {}


# Initialize FastAPI app
app = FastAPI(
    title="Steel Energy Consumption Prediction API",
    description="REST API for predicting steel industry energy consumption using Cubist regression model",
    version="1.0.0",
    lifespan=lifespan
)


def input_to_dataframe(input_data: PredictionInput) -> pd.DataFrame:
    """
    Convert PredictionInput to pandas DataFrame with correct column names.

    Args:
        input_data: Pydantic model instance

    Returns:
        DataFrame with single row
    """
    # Convert to dict and handle alias for CO2(tCO2)
    data = input_data.model_dump(by_alias=True)

    # Create DataFrame with exact column names expected by model
    # Note: Add dummy 'date' column as model expects it (used for temporal split, not as feature)
    df = pd.DataFrame([{
        'date': pd.Timestamp('2024-01-01'),  # Dummy date, not used by model
        'Lagging_Current_Reactive.Power_kVarh': data['Lagging_Current_Reactive_Power_kVarh'],
        'Leading_Current_Reactive_Power_kVarh': data['Leading_Current_Reactive_Power_kVarh'],
        'CO2(tCO2)': data['CO2(tCO2)'],
        'Lagging_Current_Power_Factor': data['Lagging_Current_Power_Factor'],
        'Leading_Current_Power_Factor': data['Leading_Current_Power_Factor'],
        'NSM': data['NSM'],
        'Day_of_week': data['Day_of_week'],  # Enum value (e.g., "Monday")
        'WeekStatus': data['WeekStatus'],  # Enum value (e.g., "Weekday")
        'Load_Type': data['Load_Type']  # Enum value (e.g., "Medium_Load")
    }])

    return df


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Steel Energy Consumption Prediction API",
        "version": "1.0.0",
        "model": model_metadata.get("model_name", "Not loaded"),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service and model status.
    """
    is_healthy = model is not None

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        model_name=model_metadata.get("model_name") if is_healthy else None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Make a single energy consumption prediction.

    Args:
        input_data: Input features for prediction

    Returns:
        Predicted energy usage in kWh

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check service health."
        )

    try:
        # Convert input to DataFrame
        input_df = input_to_dataframe(input_data)

        # Make prediction
        prediction = model.predict(input_df)

        # Extract prediction value
        predicted_value = float(prediction[0])

        logger.info(f"Prediction made: {predicted_value:.2f} kWh")

        return PredictionResponse(Usage_kWh=predicted_value)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch energy consumption predictions.

    Args:
        request: Batch prediction request with list of input instances

    Returns:
        List of predictions corresponding to each input instance

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check service health."
        )

    try:
        # Convert all inputs to DataFrame
        input_dfs = [input_to_dataframe(instance) for instance in request.instances]
        input_df = pd.concat(input_dfs, ignore_index=True)

        # Make predictions
        predictions = model.predict(input_df)

        # Create response
        prediction_responses = [
            PredictionResponse(Usage_kWh=float(pred))
            for pred in predictions
        ]

        logger.info(f"Batch prediction made: {len(prediction_responses)} predictions")

        return BatchPredictionResponse(
            predictions=prediction_responses,
            count=len(prediction_responses)
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        Model metadata including version, parameters, and metrics

    Raises:
        HTTPException: If model is not loaded
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check service health."
        )

    return ModelInfo(**model_metadata)


if __name__ == "__main__":
    import uvicorn

    # Run the application
    # Note: Use "src.api:app" when reload=True to ensure proper module loading
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
