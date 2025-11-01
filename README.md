# Steel Industry Energy Consumption Prediction

This project aims to predict the energy consumption (`Usage_kWh`) in the steel industry using various machine learning models. The goal is to build a robust regression model that can accurately forecast energy usage based on historical data.

## Project Structure

```
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── .dvc/
├── .github/
├── .gitignore
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

## Tools and Technologies

- **Python**: The core programming language used for data processing and model training.
- **DVC (Data Version Control)**: To version control data and models.
- **MLflow**: For experiment tracking, model logging, and reproducibility.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Pandas**: For data manipulation and analysis.
- **Jupyter Notebooks**: For exploratory data analysis and experimentation.

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

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1.  **Fork the repository.**
2.  **Create a new branch:** `git checkout -b feature/your-feature-name`
3.  **Make your changes** and commit them with a descriptive message.
4.  **Push your changes** to your forked repository.
5.  **Create a pull request** to the `main` branch of the original repository.

Please make sure your code follows the project's coding standards and includes tests for new features.
