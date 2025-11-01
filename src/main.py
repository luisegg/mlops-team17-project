from errno import ESTALE
import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from cubist import Cubist
from sklearn.svm import SVR
import numpy as np
import mlflow
from mlflow.models import infer_signature
import joblib
import json
import os
import tempfile
from dotenv import load_dotenv
from train import ModelTrainer, HyperparameterManager, FeaturePreprocessor, DataSplitter, MetricsCalculator

TARGET = "Usage_kWh"

# Model configurations - will be loaded from params.yaml
TARGET = "Usage_kWh"

# Model configurations - will be loaded from params.yaml
MODEL_CONFIGS = {
    "LinearRegression": {
        "name": "LinearRegression",
        "estimator": LinearRegression,
        "experiment_name": "LinearRegression"
    },
    "KNN": {
        "name": "KNNRegression",
        "estimator": KNeighborsRegressor,
        "experiment_name": "KNNRegression"
    },
    "CART": {
        "name": "CARTRegression",
        "estimator": DecisionTreeRegressor,
        "experiment_name": "CARTRegression"
    },
    "RandomForest": {
        "name": "RandomForestRegression",
        "estimator": RandomForestRegressor,
        "experiment_name": "RandomForestRegression"
    },
    "Cubist": {
        "name": "CubistRegression",
        "estimator": Cubist,
        "experiment_name": "CubistRegression"
    },
    "SVM": {
        "name": "SVMRegression",
        "estimator": SVR,
        "experiment_name": "SVMRegression"
    }
}

# Load environment variables from .env located at project root
# This prevents committing sensitive values to version control
load_dotenv()

# Read tracking configuration from environment variables
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow_tracking_username = os.getenv('MLFLOW_TRACKING_USERNAME')
mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD')


# Set process env for MLflow and clients that read from env vars
os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password

def main():
    """Main function that runs all regression models in a parent experiment."""

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    # Initialize hyperparameter manager
    hyperparameter_manager = HyperparameterManager()
    params = hyperparameter_manager.params
    data_path = params["preprocess"]["output"]
    split_cfg = params.get("split", {})
    train_cfg = params.get("train", {})
    test_size = split_cfg.get("test_size", 0.2)
    
    # Get list of models to run (default: all models)
    models_to_run = train_cfg.get("models_to_run", list(MODEL_CONFIGS.keys()))
    
    # Print combinations summary
    hyperparameter_manager.print_combinations_summary(models_to_run)
    
    # 1 Load the data
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 2 Split the data
    data_splitter = DataSplitter(target_column=TARGET)
    X_train, X_test, y_train, y_test = data_splitter.split_data(df=df, test_size=test_size)

    # 3 Build preprocessor
    feature_preprocessor = FeaturePreprocessor()
    pre = feature_preprocessor.build_preprocessor()

    # 4 Create parent experiment
    parent_experiment_name = "Steel Energy Regression Comparison"
    mlflow.set_experiment(parent_experiment_name)
    
    # 5 Train all models in the parent experiment
    metrics_calculator = MetricsCalculator()
    model_trainer = ModelTrainer(metrics_calculator=metrics_calculator)
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"Starting parent experiment: {parent_experiment_name}")
    print(f"Models to run: {', '.join(models_to_run)}")
    print(f"{'='*60}\n")
    
    with mlflow.start_run(run_name="Parent Experiment"):
        # Log parent experiment parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("models_count", len(models_to_run))
        mlflow.log_param("models_list", models_to_run)
        
        # Train each model type
        for model_name in models_to_run:
            print(f"\n--- Training {model_name} ---")
            
            try:
                model_config = model_trainer.get_model_config(model_name, hyperparameter_manager)
                
                # Train model (returns all models now)
                all_trained_models = model_trainer.train_model(
                    model_config=model_config,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    preprocessor=pre
                )
                
                # Store all results for this model type
                all_results[model_name] = {
                    'models': all_trained_models,  # All models with their metrics
                    'config': model_config
                }
                
                # Find best model for summary display
                #best_result = min(all_trained_models, key=lambda x: x['metrics']['RMSE'])
                
                print(f"✓ {model_name} completed - Trained {len(all_trained_models)} models")
                #print(f"  Best RMSE: {best_result['metrics']['RMSE']:.3f}")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                all_results[model_name] = None
        
        # 6 Log summary results in parent experiment (INSIDE the parent run context)
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        successful_models = {k: v for k, v in all_results.items() if v is not None}
        
        if successful_models:
            # Extract best metrics and params from each model type for comparison and rankings
            model_metrics = {}
            model_params = {}
            for name, result in successful_models.items():
                # Get the best model from all trained models for this algorithm
                best_result = min(result['models'], key=lambda x: x['metrics']['RMSE'])
                model_metrics[name] = best_result['metrics']
                model_params[name] = best_result['params']
            
            # Get best model for each criterion
            best_rmse_model, best_rmse_metrics = metrics_calculator.get_best_model(model_metrics, criterion='RMSE')
            best_mae_model, best_mae_metrics = metrics_calculator.get_best_model(model_metrics, criterion='MAE')
            best_r2_model, best_r2_metrics = metrics_calculator.get_best_model(model_metrics, criterion='R2')
            best_cv_model, best_cv_metrics = metrics_calculator.get_best_model(model_metrics, criterion='CV')
            
            print(f"BEST MODELS BY CRITERION:")
            print(f"  🏆 Best RMSE: {best_rmse_model} (RMSE: {best_rmse_metrics['RMSE']:.3f})")
            print(f"  🏆 Best MAE:  {best_mae_model} (MAE: {best_mae_metrics['MAE']:.3f})")
            print(f"  🏆 Best R2:   {best_r2_model} (R2: {best_r2_metrics['R2']:.3f})")
            print(f"  🏆 Best CV:   {best_cv_model} (CV: {best_cv_metrics['CV']:.3f}%)")
            
            # Print individual rankings
            metrics_calculator.print_individual_rankings(model_metrics)
            
            # Create and save all summary DataFrames as artifacts
            metrics_calculator.save_summary_artifacts(successful_models, model_metrics, model_params)
            
            # Log best models for each criterion
            mlflow.log_param("best_RMSE_model", best_rmse_model)
            mlflow.log_param("best_MAE_model", best_mae_model)
            mlflow.log_param("best_R2_model", best_r2_model)
            mlflow.log_param("best_CV_model", best_cv_model)
            
            # Log best metrics
            mlflow.log_metric("best_RMSE_value", best_rmse_metrics['RMSE'])
            mlflow.log_metric("best_MAE_value", best_mae_metrics['MAE'])
            mlflow.log_metric("best_R2_value", best_r2_metrics['R2'])
            mlflow.log_metric("best_CV_value", best_cv_metrics['CV'])
            
            # Log individual model results (best of each algorithm)
            for model_name, metrics in model_metrics.items():
                mlflow.log_metric(f"{model_name}_RMSE", metrics['RMSE'])
                mlflow.log_metric(f"{model_name}_MAE", metrics['MAE'])
                mlflow.log_metric(f"{model_name}_R2", metrics['R2'])
                mlflow.log_metric(f"{model_name}_CV", metrics['CV'])
        
        print(f"\nTotal models trained: {len(successful_models)}/{len(models_to_run)}")
        print(f"{'='*60}")
    
    return all_results



if __name__ == "__main__":
    main()