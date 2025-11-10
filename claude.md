# Steel Energy Consumption Prediction API

FastAPI-based REST service for predicting steel industry energy consumption using a Cubist regression model stored in MLflow Model Registry.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Registration](#model-registration)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Input Features](#input-features)

---

## Prerequisites

Before running the API, ensure you have:

1. **Python 3.8+** installed
2. **MLflow Model Registry** access to DagsHub
3. **Model registered** in MLflow with name `steel-energy-cubist`
4. **Environment variables** configured in `.env` file:
   ```
   MLFLOW_TRACKING_URI=https://dagshub.com/luisegg/mlops-team17-project.mlflow
   MLFLOW_TRACKING_USERNAME=your_username
   MLFLOW_TRACKING_PASSWORD=your_password
   ```

---

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository_url>
   cd mlops-team17-project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Model Registration

The API expects a model named `steel-energy-cubist` to be registered in MLflow Model Registry with stage `Production`.

### Current Model Details

- **Model Name**: `steel-energy-cubist`
- **Version**: `1`
- **Stage**: `Production`
- **Model URI**: `models:/steel-energy-cubist/Production` (or `models:/steel-energy-cubist/1`)
- **Run ID**: `a2d00acf3383409a9bb35b1482e0e4d4`
- **Hyperparameters**: n_committees=5, n_rules=100
- **Metrics**: RMSE=5.65, MAE=0.46, R2=0.968

### Registering the Model Manually

1. **Find your best Cubist model run** in MLflow UI:
   - Go to https://dagshub.com/luisegg/mlops-team17-project.mlflow
   - Navigate to "CubistRegression" experiment
   - Find the run with `n_committees=5` and `n_rules=100`
   - Note the run ID

2. **Register the model** using MLflow UI:
   - Click on the run
   - Scroll to "Artifacts" → "model"
   - Click "Register Model"
   - Model name: `steel-energy-cubist`
   - Click "Register"

3. **Transition to Production**:
   - Go to "Models" tab
   - Click on `steel-energy-cubist`
   - Click on the version you registered
   - Click "Stage" → "Transition to Production"

Alternatively, you can register programmatically:

```python
import mlflow
from dotenv import load_dotenv
import os

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Replace with your actual run_id
run_id = "your_run_id_here"
model_uri = f"runs:/{run_id}/model"

# Register model
model_name = "steel-energy-cubist"
mlflow.register_model(model_uri, model_name)

# Transition to Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=1,  # Adjust if needed
    stage="Production"
)
```

---

## Running the API

### Option 1: Using Uvicorn directly
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Using Python directly
```bash
python src/api/main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## API Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns API information and available endpoints

### 2. Health Check
- **GET** `/health`
- Check if service is running and model is loaded
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_name": "steel-energy-cubist"
  }
  ```

### 3. Single Prediction
- **POST** `/predict`
- Make a single energy consumption prediction
- **Request Body**: `PredictionInput` (see [Input Features](#input-features))
- **Response**:
  ```json
  {
    "Usage_kWh": 123.45
  }
  ```

### 4. Batch Prediction
- **POST** `/predict/batch`
- Make multiple predictions at once
- **Request Body**:
  ```json
  {
    "instances": [
      { /* PredictionInput 1 */ },
      { /* PredictionInput 2 */ }
    ]
  }
  ```
- **Response**:
  ```json
  {
    "predictions": [
      {"Usage_kWh": 123.45},
      {"Usage_kWh": 234.56}
    ],
    "count": 2
  }
  ```

### 5. Model Information
- **GET** `/model/info`
- Get metadata about the loaded model
- **Response**:
  ```json
  {
    "model_name": "steel-energy-cubist",
    "model_version": "1",
    "model_uri": "models:/steel-energy-cubist/Production",
    "run_id": "abc123...",
    "parameters": {
      "n_committees": "5",
      "n_rules": "100"
    },
    "metrics": {
      "rmse": 12.34,
      "mae": 10.12,
      "r2": 0.95
    },
    "model_type": "Cubist"
  }
  ```

---

## Usage Examples

### Example 1: Health Check (curl)
```bash
curl http://localhost:8000/health
```

### Example 2: Single Prediction (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Lagging_Current_Reactive_Power_kVarh": 5.0,
    "Leading_Current_Reactive_Power_kVarh": 3.5,
    "CO2(tCO2)": 0.05,
    "Lagging_Current_Power_Factor": 0.92,
    "Leading_Current_Power_Factor": 0.88,
    "NSM": 43200,
    "Day_of_week": "Monday",
    "WeekStatus": "Weekday",
    "Load_Type": "Medium_Load"
  }'
```

### Example 3: Single Prediction (Python)
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Lagging_Current_Reactive_Power_kVarh": 5.0,
    "Leading_Current_Reactive_Power_kVarh": 3.5,
    "CO2(tCO2)": 0.05,
    "Lagging_Current_Power_Factor": 0.92,
    "Leading_Current_Power_Factor": 0.88,
    "NSM": 43200,
    "Day_of_week": "Monday",
    "WeekStatus": "Weekday",
    "Load_Type": "Medium_Load"
}

response = requests.post(url, json=data)
print(response.json())
# Output: {"Usage_kWh": 123.45}
```

### Example 4: Batch Prediction (Python)
```python
import requests

url = "http://localhost:8000/predict/batch"
data = {
    "instances": [
        {
            "Lagging_Current_Reactive_Power_kVarh": 5.0,
            "Leading_Current_Reactive_Power_kVarh": 3.5,
            "CO2(tCO2)": 0.05,
            "Lagging_Current_Power_Factor": 0.92,
            "Leading_Current_Power_Factor": 0.88,
            "NSM": 43200,
            "Day_of_week": "Monday",
            "WeekStatus": "Weekday",
            "Load_Type": "Medium_Load"
        },
        {
            "Lagging_Current_Reactive_Power_kVarh": 7.0,
            "Leading_Current_Reactive_Power_kVarh": 4.0,
            "CO2(tCO2)": 0.08,
            "Lagging_Current_Power_Factor": 0.88,
            "Leading_Current_Power_Factor": 0.85,
            "NSM": 72000,
            "Day_of_week": "Tuesday",
            "WeekStatus": "Weekday",
            "Load_Type": "Maximum_Load"
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
# Output: {"predictions": [{"Usage_kWh": 123.45}, {"Usage_kWh": 234.56}], "count": 2}
```

### Example 5: Get Model Info (Python)
```python
import requests

url = "http://localhost:8000/model/info"
response = requests.get(url)
print(response.json())
```

---

## Input Features

All 9 features must be provided for each prediction:

### Numeric Features

1. **Lagging_Current_Reactive_Power_kVarh** (float)
   - Lagging current reactive power in kVArh
   - Example: `5.0`

2. **Leading_Current_Reactive_Power_kVarh** (float)
   - Leading current reactive power in kVArh
   - Example: `3.5`

3. **CO2(tCO2)** (float)
   - CO2 emissions in tons
   - Example: `0.05`

4. **Lagging_Current_Power_Factor** (float)
   - Lagging current power factor
   - Range: `0.0` to `1.0`
   - Example: `0.92`

5. **Leading_Current_Power_Factor** (float)
   - Leading current power factor
   - Range: `0.0` to `1.0`
   - Example: `0.88`

6. **NSM** (integer)
   - Number of seconds from midnight
   - Range: `0` to `86399` (86400 seconds in a day)
   - Example: `43200` (12:00 PM)

### Categorical Features

7. **Day_of_week** (string)
   - Valid values: `Monday`, `Tuesday`, `Wednesday`, `Thursday`, `Friday`, `Saturday`, `Sunday`
   - Example: `"Monday"`

8. **WeekStatus** (string)
   - Valid values: `Weekday`, `Weekend`
   - Example: `"Weekday"`

9. **Load_Type** (string)
   - Valid values: `Light_Load`, `Medium_Load`, `Maximum_Load`
   - Example: `"Medium_Load"`

---

## Error Handling

The API returns appropriate HTTP status codes:

- **200 OK**: Successful prediction
- **422 Unprocessable Entity**: Invalid input (validation error)
- **500 Internal Server Error**: Prediction failed
- **503 Service Unavailable**: Model not loaded

Example error response:
```json
{
  "detail": "Model not loaded. Please check service health."
}
```

---

## Testing the API

You can test the API using the interactive documentation at http://localhost:8000/docs, which provides:
- Interactive form for each endpoint
- Request/response schemas
- Try it out functionality
- Example values

---

## Troubleshooting

### Issue: Model not loading on startup
- **Solution**: Check that the model is registered in MLflow as `steel-energy-cubist` with stage `Production`
- Verify `.env` file contains correct DagsHub credentials
- Check logs for detailed error messages

### Issue: Connection error to MLflow
- **Solution**: Verify `MLFLOW_TRACKING_URI` in `.env` file
- Check internet connection
- Verify DagsHub credentials are correct

### Issue: Prediction fails with 500 error
- **Solution**: Check input feature names and types match exactly
- Verify all 9 required features are provided
- Check API logs for detailed error messages

---

## Project Structure

```
mlops-team17-project/
├── src/
│   ├── api.py              # FastAPI application
│   ├── schemas.py          # Pydantic models
│   ├── train.py            # Model training script
│   └── ...
├── .env                    # Environment variables (MLflow credentials)
├── requirements.txt        # Python dependencies
└── README_API.md          # This file
```

---

## License

This project is part of the MLOps Team 17 project for steel energy consumption prediction.
