# MLOPS Equipo 17

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project aims to predict the energy consumption (`Usage_kWh`) in the steel industry using various machine learning models. The goal is to build a robust regression model that can accurately forecast energy usage based on historical data.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Fase 2 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes Fase 2 a Python module
    ├── train.py                <- Code to train models
    ├── evaluate.py             <- Code to evaluate models
    ├── preprocess.py           <- Code to preprocess the data
    ├── main.py                 <- Main script to run the pipeline
    └── api/                    <- FastAPI service module
        ├── __init__.py         <- Makes api a Python module
        ├── main.py             <- FastAPI application and endpoints
        ├── schemas.py          <- Pydantic schemas for API validation
        └── check_model.py      <- Utility to check MLflow model registration

```

## Tools and Technologies

- **Python**: The core programming language used for data processing and model training.
- **DVC (Data Version Control)**: To version control data and models.
- **MLflow**: For experiment tracking, model logging, and reproducibility.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Pandas**: For data manipulation and analysis.
- **Jupyter Notebooks**: For exploratory data analysis and experimentation.
- **FastAPI**: For serving model predictions via REST API.
- **Cubist**: Rule-based regression algorithm for energy consumption prediction.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd mlops-team17-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**

    The training script requires MLflow credentials, which are loaded from a `.env` file at the project's root. This file is not committed to version control for security reasons. You will need to create a `.env` file in the root of the project with the following content:

    ```
    MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
    MLFLOW_TRACKING_USERNAME=your_username
    MLFLOW_TRACKING_PASSWORD=your_password
    ```
    Replace the placeholder values with your actual MLflow credentials.

5.  **Pull the data and models versioned with DVC:**
    ```bash
    dvc pull
    ```

## Usage

### Data Preprocessing

To process the raw data, run the preprocessing script:

```bash
python src/preprocess.py
```

### Model Training

To train the models specified in `params.yaml`, run the training script:

```bash
python src/train.py
```

This will train the models, log experiments to MLflow, and save the best model.

### Model Deployment (FastAPI)

The best performing model (Cubist with n_committees=5, n_rules=100) is deployed as a REST API service.

**Deployed Model Details:**
- **Model Name**: `steel-energy-cubist`
- **Version**: `1`
- **Stage**: `Production`
- **Model URI**: `models:/steel-energy-cubist/Production` (or `models:/steel-energy-cubist/1`)
- **Run ID**: `a2d00acf3383409a9bb35b1482e0e4d4`
- **Performance**: RMSE=5.65, MAE=0.46, R²=0.968

**Starting the API:**

```bash
# Start the FastAPI service
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model metadata
- `GET /docs` - Interactive API documentation

**Example Request:**

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

For complete API documentation, see `claude.md`.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1.  **Fork the repository.**
2.  **Create a new branch:** `git checkout -b feature/your-feature-name`
3.  **Make your changes** and commit them with a descriptive message.
4.  **Push your changes** to your forked repository.
5.  **Create a pull request** to the `main` branch of the original repository.

Please make sure your code follows the project's coding standards and includes tests for new features.

--------

