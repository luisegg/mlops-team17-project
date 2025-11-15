from errno import ESTALE
import pandas as pd
from pathlib import Path
import yaml
import numpy as np
import random
import os 

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from cubist import Cubist
import mlflow
from mlflow.models import infer_signature
import joblib
import json
import tempfile
from dotenv import load_dotenv


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

#Desactivation of multithreading for MLflow to avoid issues in some environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

class HyperparameterManager:
    """
    Class to manage hyperparameter configurations and generate parameter combinations.
    """
    
    def __init__(self, params_path="params.yaml"):
        """
        Initialize the HyperparameterManager.
        
        Args:
            params_path (str): Path to the parameters YAML file
        """
        self.params_path = params_path
        self.params = self.load_params()
    
    def load_params(self):
        """
        Load parameters from YAML file.
        
        Returns:
            dict: Parameters dictionary
        """
        if Path(self.params_path).exists():
            with open(self.params_path, "r") as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(f"{self.params_path} not found")
    
    def generate_param_combinations(self, model_name):
        """
        Generate parameter combinations for a model from params.yaml.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            list: List of parameter dictionaries
        """
        import itertools
        
        if model_name == "LinearRegression":
            linear_params = self.params.get("linear_regression", {})
            fit_intercept_options = linear_params.get("fit_intercept", [True, False])
            return [{"fit_intercept": option} for option in fit_intercept_options]
        
        elif model_name == "KNN":
            knn_params = self.params.get("knn", {})
            k_values = knn_params.get("k", [3, 5, 7, 11])
            weights_options = knn_params.get("weights", ["uniform", "distance"])
            p_values = knn_params.get("p", [1, 2])
            
            combinations = []
            for k in k_values:
                for weights in weights_options:
                    for p in p_values:
                        combinations.append({
                            "n_neighbors": k,
                            "weights": weights,
                            "p": p
                        })
            return combinations
        
        elif model_name == "CART":
            cart_params = self.params.get("cart", {})
            
            # Get parameter ranges
            ccp_alpha = cart_params.get("ccp_alpha", [0.0, 0.1])
            max_depth = cart_params.get("max_depth", [5, 10])
            min_samples_split = cart_params.get("min_samples_split", [2, 5])
            min_samples_leaf = cart_params.get("min_samples_leaf", [1, 2])
            max_features = cart_params.get("max_features", ["sqrt"])
            criterion = cart_params.get("criterion", ["squared_error"])
            
            # Generate combinations
            combinations = []
            for ccp in ccp_alpha:
                for max_d in max_depth:
                    for min_split in min_samples_split:
                        for min_leaf in min_samples_leaf:
                            for max_feat in max_features:
                                for crit in criterion:
                                    combinations.append({
                                        "ccp_alpha": ccp,
                                        "max_depth": max_d,
                                        "min_samples_split": min_split,
                                        "min_samples_leaf": min_leaf,
                                        "max_features": max_feat,
                                        "criterion": crit,
                                        "random_state": 42,
                                        "splitter": "best"
                                    })
            return combinations
        
        elif model_name == "RandomForest":
            rf_params = self.params.get("random_forest", {})
            
            # Get parameter ranges
            n_estimators = rf_params.get("n_estimators", [100, 200])
            max_depth = rf_params.get("max_depth", [10, 15])
            min_samples_split = rf_params.get("min_samples_split", [2, 5])
            min_samples_leaf = rf_params.get("min_samples_leaf", [1, 2])
            max_features = rf_params.get("max_features", ["sqrt"])
            criterion = rf_params.get("criterion", ["squared_error"])
            bootstrap = rf_params.get("bootstrap", [True])
            
            # Generate combinations
            combinations = []
            for n_est in n_estimators:
                for max_d in max_depth:
                    for min_split in min_samples_split:
                        for min_leaf in min_samples_leaf:
                            for max_feat in max_features:
                                for crit in criterion:
                                    for boot in bootstrap:
                                        combinations.append({
                                            "n_estimators": n_est,
                                            "max_depth": max_d,
                                            "min_samples_split": min_split,
                                            "min_samples_leaf": min_leaf,
                                            "max_features": max_feat,
                                            "criterion": crit,
                                            "bootstrap": boot,
                                            "random_state": 42,
                                            "n_jobs": 1
                                        })
            return combinations
        
        elif model_name == "Cubist":
            cubist_params = self.params.get("cubist", {})
            n_committees = cubist_params.get("n_committees", [1, 5, 10, 20])
            n_rules = cubist_params.get("n_rules", [10, 50, 100])
            
            combinations = []
            for n_comm in n_committees:
                for n_rule in n_rules:
                    combinations.append({
                        "n_committees": n_comm,
                        "n_rules": n_rule
                    })
            return combinations
        
        elif model_name == "SVM":
            param_grid = self.params.get("svm", {})
            keys, values = zip(*param_grid.items())
            combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

            return combos
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def get_model_config(self, model_name):
        """
        Get model configuration by name with parameters from params.yaml.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration with generated parameters
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS")
        
        config = MODEL_CONFIGS[model_name].copy()
        config["params"] = self.generate_param_combinations(model_name)
        return config
    
    def get_total_combinations(self, model_name):
        """
        Get the total number of parameter combinations for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            int: Number of parameter combinations
        """
        return len(self.generate_param_combinations(model_name))
    
    def print_combinations_summary(self, models_to_run):
        """
        Print a summary of parameter combinations for all models.
        
        Args:
            models_to_run (list): List of model names to run
        """
        print(f"\n{'='*60}")
        print("HYPERPARAMETER COMBINATIONS SUMMARY")
        print(f"{'='*60}")
        
        total_combinations = 0
        for model_name in models_to_run:
            combinations = self.get_total_combinations(model_name)
            total_combinations += combinations
            print(f"{model_name:15}: {combinations:3d} combinations")
        
        print(f"{'='*60}")
        print(f"{'TOTAL':15}: {total_combinations:3d} combinations")
        print(f"{'='*60}")



class MetricsCalculator:
    """Class responsible for calculating and evaluating model metrics."""
    
    def __init__(self):
        """Initialize MetricsCalculator."""
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate RMSE, MAE, R2, and CV metrics for regression models.
    
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
                dict: Dictionary containing RMSE, MAE, R2, and CV metrics
        """
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        cv = self.calculate_cv_percent(y_true, y_pred)
            
        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "CV": cv}
        return metrics
    
    def calculate_cv_percent(self, y_true, y_pred):
        """
        Calculate coefficient of variation percentage.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: CV percentage
        """
        rmse = root_mean_squared_error(y_true, y_pred)
        return 100.0 * rmse / np.mean(y_true)
    
    def print_metrics(self, metrics, model_name="Model"):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics (dict): Dictionary of metrics
            model_name (str): Name of the model
        """
        print(f"\n[{model_name}] Metrics:")
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  MAE:  {metrics['MAE']:.3f}")
        print(f"  R2:   {metrics['R2']:.3f}")
        print(f"  CV:   {metrics['CV']:.3f}%")
    
    def compare_models(self, model_results):
        """
        Compare multiple models and return ranking.
        
        Args:
            model_results (dict): Dictionary with model names as keys and metrics as values
            
        Returns:
            list: Sorted list of tuples (model_name, metrics) by RMSE
        """
        return sorted(model_results.items(), key=lambda x: x[1]['RMSE'])
    
    def get_best_model(self, model_results, criterion='RMSE'):
        """
        Get the best model based on specified criterion.
        
        Args:
            model_results (dict): Dictionary with model names as keys and metrics as values
            criterion (str): 'RMSE', 'MAE', 'R2', 'CV', or 'multi_criteria'
            
        Returns:
            tuple: (best_model_name, best_metrics)
        """
        if criterion == 'RMSE':
            return min(model_results.items(), key=lambda x: x[1]['RMSE'])
        elif criterion == 'MAE':
            return min(model_results.items(), key=lambda x: x[1]['MAE'])
        elif criterion == 'R2':
            return max(model_results.items(), key=lambda x: x[1]['R2'])
        elif criterion == 'CV':
            return min(model_results.items(), key=lambda x: x[1]['CV'])
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def print_individual_rankings(self, model_results):
        """
        Print individual rankings for each metric.
        
        Args:
            model_results (dict): Dictionary with model names as keys and metrics as values
        """
        print(f"\n{'='*60}")
        print("INDIVIDUAL METRIC RANKINGS")
        print(f"{'='*60}")
        
        # RMSE Ranking (lower is better)
        rmse_ranking = sorted(model_results.items(), key=lambda x: x[1]['RMSE'])
        print(f"\nRMSE Ranking (lower is better):")
        for i, (model_name, metrics) in enumerate(rmse_ranking, 1):
            print(f"  {i}. {model_name}: {metrics['RMSE']:.3f}")
        
        # MAE Ranking (lower is better)
        mae_ranking = sorted(model_results.items(), key=lambda x: x[1]['MAE'])
        print(f"\nMAE Ranking (lower is better):")
        for i, (model_name, metrics) in enumerate(mae_ranking, 1):
            print(f"  {i}. {model_name}: {metrics['MAE']:.3f}")
        
        # R2 Ranking (higher is better)
        r2_ranking = sorted(model_results.items(), key=lambda x: x[1]['R2'], reverse=True)
        print(f"\nR2 Ranking (higher is better):")
        for i, (model_name, metrics) in enumerate(r2_ranking, 1):
            print(f"  {i}. {model_name}: {metrics['R2']:.3f}")
        
        # CV Ranking (lower is better)
        cv_ranking = sorted(model_results.items(), key=lambda x: x[1]['CV'])
        print(f"\nCV Ranking (lower is better):")
        for i, (model_name, metrics) in enumerate(cv_ranking, 1):
            print(f"  {i}. {model_name}: {metrics['CV']:.3f}%")
        
        print(f"{'='*60}")
    
    def _params_to_string(self, params_dict):
        """
        Convert parameters dictionary to comma-separated string.
        
        Args:
            params_dict (dict): Dictionary of parameters
            
        Returns:
            str: Comma-separated string representation
        """
        if not params_dict:
            return "default"
        return ", ".join([f"{k}={v}" for k, v in params_dict.items()])
    
    def create_all_models_summary_df(self, successful_models):
        """
        Create DataFrame with all models (all child runs) and their metrics.
        
        Args:
            successful_models (dict): Dictionary with model results
            
        Returns:
            pd.DataFrame: DataFrame with all models and metrics
        """
        summary_data = []
        for model_name, result in successful_models.items():
            # Iterate through ALL models for this algorithm (not just the best)
            for model_result in result['models']:
                summary_data.append({
                    'Model': model_name,
                    'Hyperparameters': self._params_to_string(model_result['params']),
                    'RMSE': model_result['metrics']['RMSE'],
                    'MAE': model_result['metrics']['MAE'],
                    'R2': model_result['metrics']['R2'],
                    'CV': model_result['metrics']['CV']
                })
        return pd.DataFrame(summary_data)
    
    def create_best_models_by_criterion_df(self, model_metrics, model_params):
        """
        Create DataFrame with best model for each criterion.
        
        Args:
            model_metrics (dict): Dictionary with model names as keys and metrics as values
            model_params (dict): Dictionary with model names as keys and params as values
            
        Returns:
            pd.DataFrame: DataFrame with best models by criterion
        """
        best_rmse_model, best_rmse_metrics = self.get_best_model(model_metrics, criterion='RMSE')
        best_mae_model, best_mae_metrics = self.get_best_model(model_metrics, criterion='MAE')
        best_r2_model, best_r2_metrics = self.get_best_model(model_metrics, criterion='R2')
        best_cv_model, best_cv_metrics = self.get_best_model(model_metrics, criterion='CV')
        
        best_models_data = [
            {'Criterion': 'RMSE', 'Best_Model': best_rmse_model, 'Hyperparameters': self._params_to_string(model_params[best_rmse_model]), 'Value': best_rmse_metrics['RMSE']},
            {'Criterion': 'MAE', 'Best_Model': best_mae_model, 'Hyperparameters': self._params_to_string(model_params[best_mae_model]), 'Value': best_mae_metrics['MAE']},
            {'Criterion': 'R2', 'Best_Model': best_r2_model, 'Hyperparameters': self._params_to_string(model_params[best_r2_model]), 'Value': best_r2_metrics['R2']},
            {'Criterion': 'CV', 'Best_Model': best_cv_model, 'Hyperparameters': self._params_to_string(model_params[best_cv_model]), 'Value': best_cv_metrics['CV']}
        ]
        return pd.DataFrame(best_models_data)
    
    def create_rankings_dfs(self, model_metrics, model_params):
        """
        Create ranking DataFrames for each metric.
        
        Args:
            model_metrics (dict): Dictionary with model names as keys and metrics as values
            model_params (dict): Dictionary with model names as keys and params as values
            
        Returns:
            dict: Dictionary with ranking DataFrames for each metric
        """
        # RMSE Ranking (lower is better)
        rmse_ranking = sorted(model_metrics.items(), key=lambda x: x[1]['RMSE'])
        rmse_ranking_df = pd.DataFrame([
            {'Rank': i+1, 'Model': model_name, 'Hyperparameters': self._params_to_string(model_params[model_name]), 'RMSE': metrics['RMSE']}
            for i, (model_name, metrics) in enumerate(rmse_ranking)
        ])
        
        # MAE Ranking (lower is better)
        mae_ranking = sorted(model_metrics.items(), key=lambda x: x[1]['MAE'])
        mae_ranking_df = pd.DataFrame([
            {'Rank': i+1, 'Model': model_name, 'Hyperparameters': self._params_to_string(model_params[model_name]), 'MAE': metrics['MAE']}
            for i, (model_name, metrics) in enumerate(mae_ranking)
        ])
        
        # R2 Ranking (higher is better)
        r2_ranking = sorted(model_metrics.items(), key=lambda x: x[1]['R2'], reverse=True)
        r2_ranking_df = pd.DataFrame([
            {'Rank': i+1, 'Model': model_name, 'Hyperparameters': self._params_to_string(model_params[model_name]), 'R2': metrics['R2']}
            for i, (model_name, metrics) in enumerate(r2_ranking)
        ])
        
        # CV Ranking (lower is better)
        cv_ranking = sorted(model_metrics.items(), key=lambda x: x[1]['CV'])
        cv_ranking_df = pd.DataFrame([
            {'Rank': i+1, 'Model': model_name, 'Hyperparameters': self._params_to_string(model_params[model_name]), 'CV': metrics['CV']}
            for i, (model_name, metrics) in enumerate(cv_ranking)
        ])
        
        return {
            'rmse': rmse_ranking_df,
            'mae': mae_ranking_df,
            'r2': r2_ranking_df,
            'cv': cv_ranking_df
        }
    
    def save_summary_artifacts(self, successful_models, model_metrics, model_params):
        """
        Create and save all summary DataFrames as MLflow artifacts.
        
        Args:
            successful_models (dict): Dictionary with all model results
            model_metrics (dict): Dictionary with model names as keys and metrics as values
            model_params (dict): Dictionary with model names as keys and params as values
        """
        # Create DataFrames
        summary_df = self.create_all_models_summary_df(successful_models)
        best_models_df = self.create_best_models_by_criterion_df(model_metrics, model_params)
        rankings_dfs = self.create_rankings_dfs(model_metrics, model_params)
        
        # Save all DataFrames as artifacts with descriptive names
        artifacts_to_save = [
            (summary_df, 'models_metrics_summary.csv', 'summary'),
            (best_models_df, 'best_models_by_criterion.csv', 'summary'),
            (rankings_dfs['rmse'], 'rmse_ranking.csv', 'rankings'),
            (rankings_dfs['mae'], 'mae_ranking.csv', 'rankings'),
            (rankings_dfs['r2'], 'r2_ranking.csv', 'rankings'),
            (rankings_dfs['cv'], 'cv_ranking.csv', 'rankings')
        ]
        
        for df, filename, artifact_path in artifacts_to_save:
            temp_file = os.path.join(tempfile.gettempdir(), filename)
            df.to_csv(temp_file, index=False)
            mlflow.log_artifact(temp_file, artifact_path)
            os.remove(temp_file)

class DataSplitter:
    """Class responsible for splitting data into train and test sets."""
    
    def __init__(self, target_column="Usage_kWh"):
        """
        Initialize DataSplitter.
        
        Args:
            target_column (str): Name of the target column
        """
        self.target_column = target_column
    
    def split_data(self, df, test_size, shuffle=False):
        """
        Split data into train and test sets.
        
        Args:
            df (pd.DataFrame): DataFrame with data to split
            test_size (float): Proportion of data to use for testing
            shuffle (bool): Whether to shuffle data before splitting
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Prepare features and target
        X, y = self.prepare_features_target(df)
        
        # Sort by date if not shuffling (temporal split)
        if not shuffle:
            X, y = self.sort_by_date(df, X, y)

        # Split with scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=shuffle, random_state=42
        )

            # Print split information
        print(f"[split] test_size={test_size}")
        print(f"[split] X_train: {X_train.shape} | X_test: {X_test.shape}")
        print(f"[split] y_train: {y_train.shape} | y_test: {y_test.shape}")

        return X_train, X_test, y_train, y_test

    def prepare_features_target(self, df):
        """
        Separate features and target from DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            
        Returns:
            tuple: (X, y) where X is features and y is target
        """
        y = df[self.target_column].copy()
        X = df.drop(columns=[self.target_column]).copy()
        
        return X, y
    
    def sort_by_date(self, df, X, y):
        """
        Sort features and target by date column.
        
        Args:
            df (pd.DataFrame): Original DataFrame with date column
            X (pd.DataFrame): Features DataFrame
            y (pd.Series): Target Series
            
        Returns:
            tuple: (X_sorted, y_sorted)
        """
        # Get sort order based on date column
        order = df["date"].argsort()
        
        # Sort both features and target using the same order
        X_sorted = X.iloc[order]
        y_sorted = y.iloc[order]
        
        return X_sorted, y_sorted

class FeaturePreprocessor:
    """Class responsible for building feature preprocessing pipelines."""
    
    def __init__(self):
        """Initialize FeaturePreprocessor."""
        self.numeric_features = [
            'Lagging_Current_Reactive.Power_kVarh', 
            'Leading_Current_Reactive_Power_kVarh', 
            'CO2(tCO2)', 
            'Lagging_Current_Power_Factor', 
            'Leading_Current_Power_Factor', 
            'NSM'
        ]
        
        self.categorical_features = {
            'Day_of_week': {
                'categories': [['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']],
                'drop': 'first',
                'sparse_output': False
            },
            'WeekStatus': {
                'categories': [['Weekend', 'Weekday']],
                'drop': 'first',
                'sparse_output': False
            },
            'Load_Type': {
                'categories': [['Light_Load','Medium_Load','Maximum_Load']],
                'dtype': np.int8,
                'handle_unknown': 'use_encoded_value',
                'unknown_value': -1
            }
        }
    
    def build_preprocessor(self):
        """
        Build the complete preprocessing pipeline using ColumnTransformer.
        
        Returns:
            ColumnTransformer: Complete preprocessing pipeline
        """
        transformers = []
        
        # Add numeric features preprocessing
        transformers.append(self._get_numeric_transformer())
        
        # Add categorical features preprocessing
        for feature_name, config in self.categorical_features.items():
            if feature_name == 'Load_Type':
                transformers.append(self._get_ordinal_transformer(feature_name, config))
            else:
                transformers.append(self._get_onehot_transformer(feature_name, config))
        
        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        return preprocessor
    
    def _get_numeric_transformer(self):
        """Get transformer for numeric features."""
        return (
            'numeric_cols',
            Pipeline([
                ('Standard_Scaler', StandardScaler())
            ]),
            self.numeric_features
        )
    
    def _get_onehot_transformer(self, feature_name, config):
        """Get OneHotEncoder transformer for categorical features."""
        return (
            f'{feature_name}_oh',
            Pipeline([
                ('ohe', OneHotEncoder(
                    categories=config['categories'],
                    drop=config['drop'],
                    sparse_output=config['sparse_output']
                ))
            ]),
            [feature_name]
        )
    
    def _get_ordinal_transformer(self, feature_name, config):
        """Get OrdinalEncoder transformer for ordinal features."""
        return (
            f'{feature_name}_ord',
            Pipeline([
                ('ord', OrdinalEncoder(
                    categories=config['categories'],
                    dtype=config['dtype'],
                    handle_unknown=config['handle_unknown'],
                    unknown_value=config['unknown_value']
                ))
            ]),
            [feature_name]
        )
    
    def get_numeric_features(self):
        """Get list of numeric features."""
        return self.numeric_features.copy()
    
    def get_categorical_features(self):
        """Get list of categorical features."""
        return list(self.categorical_features.keys())
    
    def get_all_features(self):
        """Get list of all features used in preprocessing."""
        return self.numeric_features + self.get_categorical_features()

class ModelTrainer:
    """Class responsible for training different ML models using configuration."""
    
    def __init__(self, metrics_calculator=None):
        """
        Initialize ModelTrainer with dependencies.
        
        Args:
            metrics_calculator: Instance of MetricsCalculator
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
    
    def train_model(self, model_config, X_train, y_train, X_test, y_test, preprocessor):
        """
        Generic method to train any model based on configuration.
        
        Args:
            model_config (dict): Model configuration from MODEL_CONFIGS
            X_train, y_train: Training data
            X_test, y_test: Test data
            preprocessor: Feature preprocessor
            
        Returns:
            list: List of dictionaries containing all trained models with their metrics and params
        """
        experiment_name = model_config.get("experiment_name", model_config["name"])
        estimator_class = model_config["estimator"]
        params_list = model_config.get("params", [{}])
        
        all_models = []  # Store all models instead of just the best
        
        # Test each parameter combination
        for params in params_list:
            estimator = estimator_class(**params)
            
            model, metrics = self._train_single_model(
                estimator=estimator,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                X_test_original=X_test,  # Pass original X_test for artifact logging
                preprocessor=preprocessor,
                experiment_name=experiment_name,
                params=params
            )
            
            # Store all models with their metrics and params
            all_models.append({
                'model': model,
                'metrics': metrics,
                'params': params
            })
        
        return all_models
    
    def _train_single_model(self, estimator, X_train, y_train, X_test, y_test, X_test_original, preprocessor, experiment_name, params):
        """
        Private method to train a single model instance.
        
        Args:
            estimator: sklearn estimator instance
            X_train, y_train: Training data
            X_test, y_test: Test data
            X_test_original: Original X_test data (before preprocessing) for artifact logging
            preprocessor: Feature preprocessor
            experiment_name (str): Name for MLflow experiment
            params (dict): Model parameters
            
        Returns:
            tuple: (pipeline, metrics)
        """
        # Create pipeline
        pipeline = Pipeline([("prep", preprocessor), ("est", estimator)])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred)
        
        # Use nested runs within the parent experiment
        with mlflow.start_run(run_name=f"{experiment_name}_{params}", nested=True):
            mlflow.log_params(params)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            
            # Create DataFrame with original features, actual target, and predictions
            predictions_df = X_test_original.copy()
            predictions_df['y_actual'] = y_test.values
            predictions_df['y_predicted'] = y_pred
            
            # Add error metrics per row
            predictions_df['residual_error'] = predictions_df['y_actual'] - predictions_df['y_predicted']
            predictions_df['percentage_error'] = (predictions_df['residual_error'] / predictions_df['y_actual']) * 100
            
            # Save DataFrame as CSV artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_path = f.name
                predictions_df.to_csv(temp_path, index=False)
            
            # Log the CSV file as artifact
            mlflow.log_artifact(temp_path, "predictions")
            
            # Clean up temporary file
            os.remove(temp_path)
            
            # Log the trained pipeline as a model artifact
            signature = infer_signature(X_test, y_pred)
            mlflow.sklearn.log_model(pipeline, "model", signature=signature, input_example=X_test.head(5))
        
        return pipeline, metrics
    
    def get_model_config(self, model_name, hyperparameter_manager):
        """
        Get model configuration by name with parameters from HyperparameterManager.
        
        Args:
            model_name (str): Name of the model
            hyperparameter_manager (HyperparameterManager): Manager for hyperparameters
            
        Returns:
            dict: Model configuration with generated parameters
        """
        return hyperparameter_manager.get_model_config(model_name)




