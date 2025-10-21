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
import numpy as np
import mlflow
from mlflow.models import infer_signature
import joblib
import json
import os
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
                                        "random_state": 42
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
                                            "n_jobs": -1
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
        
        else:
            return [{}]  # Default empty parameters
    
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


'''
def log_run_to_mlflow(experiment_name, params, metrics, model=None, run_name=None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        #if model is not None:
            #mlflow.sklearn.log_model(model, "model")
'''

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
        
        return {"RMSE": rmse, "MAE": mae, "R2": r2, "CV": cv}
    
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


'''
# Legacy functions for backward compatibility
def calculate_metrics(y_true, y_pred):
    """Legacy function - use MetricsCalculator.calculate_metrics() instead."""
    calculator = MetricsCalculator()
    return calculator.calculate_metrics(y_true, y_pred)

def cv_percent(y_true, y_pred):
    """Legacy function - use MetricsCalculator.calculate_cv_percent() instead."""
    calculator = MetricsCalculator()
    return calculator.calculate_cv_percent(y_true, y_pred)
'''

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
            X, y, test_size=test_size, shuffle=shuffle
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
    
    def __init__(self, metrics_calculator=None, mlflow_logger=None):
        """
        Initialize ModelTrainer with dependencies.
        
        Args:
            metrics_calculator: Instance of MetricsCalculator
            mlflow_logger: Instance of MLflowLogger
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.mlflow_logger = mlflow_logger
    
    def train_model(self, model_config, X_train, y_train, X_test, y_test, preprocessor):
        """
        Generic method to train any model based on configuration.
        
        Args:
            model_config (dict): Model configuration from MODEL_CONFIGS
            X_train, y_train: Training data
            X_test, y_test: Test data
            preprocessor: Feature preprocessor
            
        Returns:
            tuple: (best_model, best_metrics)
        """
        experiment_name = model_config.get("experiment_name", model_config["name"])
        estimator_class = model_config["estimator"]
        params_list = model_config.get("params", [{}])
        
        best_model = None
        best_metrics = None
        
        # Test each parameter combination
        for params in params_list:
            estimator = estimator_class(**params)
            
            model, metrics = self._train_single_model(
                estimator=estimator,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                preprocessor=preprocessor,
                experiment_name=experiment_name,
                params=params
            )
            
            # Keep track of best model
            if best_model is None or metrics['RMSE'] < best_metrics['RMSE']:
                best_model = model
                best_metrics = metrics
        
        return best_model, best_metrics
    
    def _train_single_model(self, estimator, X_train, y_train, X_test, y_test, preprocessor, experiment_name, params):
        """
        Private method to train a single model instance.
        
        Args:
            estimator: sklearn estimator instance
            X_train, y_train: Training data
            X_test, y_test: Test data
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
        
        # Log to MLflow (use current experiment context)
        if self.mlflow_logger:
            self.mlflow_logger.log_experiment(
                experiment_name=experiment_name,
                params=params,
                metrics=metrics,
                model=pipeline
            )
        else:
            # Use nested runs within the parent experiment
            with mlflow.start_run(run_name=f"{experiment_name}_{params}", nested=True):
                mlflow.log_params(params)
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
                # mlflow.sklearn.log_model(pipeline, "model")
        
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


'''
def run_linear_regression(X_train, y_train, X_test, y_test, pre):
    """LinearRegression no tiene hiperparámetros: un solo run."""
    exp = "LinearRegression"
    est = LinearRegression(fit_intercept=True)
    pipe = Pipeline([("prep", pre), ("est", est)])

    # Fit the model
    pipe.fit(X_train, y_train)

    #Predicts
    y_pred = pipe.predict(X_test)

    # Gets metrics
    metrics = calculate_metrics(y_test, y_pred)

    log_run_to_mlflow(exp, params={}, metrics=metrics, model=pipe)
'''
'''
def run_knn_regression(X_train, y_train, X_test, y_test, pre, knn_params):
    exp = "KNNRegression"
   
    k_list = knn_params.get("k", [3, 5, 7, 11])
    weights_list = knn_params.get("weights", ["uniform", "distance"])
    p_list = knn_params.get("p", [1, 2])
    metric = "minkowski"

    for k in k_list:
        for w in weights_list:
            for p in p_list:
                params = {"n_neighbors": k, "weights": w, "metric": metric, "p": p}
                est = KNeighborsRegressor(**params)
                pipe = Pipeline([("prep", pre), ("est", est)])

                # Fit the model
                pipe.fit(X_train, y_train)

                #Predicts
                y_pred = pipe.predict(X_test)

                # Gets metrics
                metrics = calculate_metrics(y_test, y_pred)

                log_run_to_mlflow(exp, params=params, metrics=metrics, model=pipe)
'''
'''
def run_cart_regression(X_train, y_train, X_test, y_test, pre):
    exp = "CARTRegression"

    max_depth_list = [None, 5]
    min_samples_split_list = [2, 5]
    min_samples_leaf_list = [1, 2, 5]
    max_features_list = [None, "sqrt", "log2"]  # nº features consideradas al dividir

    for md in max_depth_list:
        for mss in min_samples_split_list:
            for msl in min_samples_leaf_list:
                for mf in max_features_list:
                    params = {
                        "max_depth": md,
                        "min_samples_split": mss,
                        "min_samples_leaf": msl,
                        "max_features": mf,
                        "random_state": 42,
                    }

                    est = DecisionTreeRegressor(**params)
                    pipe = Pipeline([("prep", pre), ("est", est)])

                    # Entrena
                    pipe.fit(X_train, y_train)

                    # Predice
                    y_pred = pipe.predict(X_test)

                    # Métricas
                    metrics = calculate_metrics(y_test, y_pred)
                    print(f"[CART] {params} -> RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f}")
                    log_run_to_mlflow(exp, params=params, metrics=metrics, model=None)
'''
'''
def run_cart_regression2(X_train, y_train, X_test, y_test, pre, cart_params):
    exp = "CARTRegression"

    # --- 1) candidates de ccp_alpha ---
    alphas = cart_params.get("ccp_alpha", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # --- 2) probar cada alpha ---
    for alpha in alphas:
        est = DecisionTreeRegressor(ccp_alpha=float(alpha), random_state=42)
        pipe = Pipeline([("prep", pre), ("est", est)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        params  = {"ccp_alpha": float(alpha), "random_state": 42}

        # Reutiliza tu helper de MLflow
        log_run_to_mlflow(experiment_name=exp,
                          params=params, metrics=metrics, model=None)

        print(f"[CART] ccp_alpha={alpha:.6g} -> RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f} CV={metrics['CV']:.3f}")
'''
'''
def hyperparameter_tuning(X_train, y_train, pre, est, param_grid):
    pipe = Pipeline([("prep", pre), ("est", est)])
    grid_search=GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2,
                             scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    return grid_search
'''
'''
def run_linear_regression_tuning(X_train, y_train, X_test, y_test, pre):
    param_grid = {
        "est__fit_intercept": [True, False]
    }
    model = LinearRegression()
    grid_search = hyperparameter_tuning(X_train, y_train, pre, model, param_grid)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    log_run_to_mlflow(experiment_name='Algorithm Comparison',
                        params=grid_search.best_params_, metrics=metrics, model=best_model,
                        run_name='Linear Regression')
    return best_model
'''
'''
def run_knn_regression_tuning(X_train, y_train, X_test, y_test, pre):
    param_grid = {
        "est__n_neighbors": [3, 5, 7, 11],
        "est__weights": ["uniform", "distance"],
        "est__metric": ["minkowski"],
        "est__p": [1, 2]
    }
    model = KNeighborsRegressor()
    grid_search = hyperparameter_tuning(X_train, y_train, pre, model, param_grid)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    log_run_to_mlflow(experiment_name='Algorithm Comparison',
                        params=grid_search.best_params_, metrics=metrics, model=best_model,
                        run_name='KNN Regression')
    return best_model
'''
'''
def run_cart_regression2_tuning(X_train, y_train, X_test, y_test, pre):
    param_grid = {
        "est__ccp_alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    model = DecisionTreeRegressor()
    grid_search = hyperparameter_tuning(X_train, y_train, pre, model, param_grid)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    log_run_to_mlflow(experiment_name='Algorithm Comparison',
                        params=grid_search.best_params_, metrics=metrics, model=best_model,
                        run_name='CART Regression')
    return best_model
'''
'''
def run_random_forest_regression(X_train, y_train, X_test, y_test, pre):
    exp = "RandomForestRegression"

    params = {
        "n_estimators": 100,           # Número de árboles
        "criterion": "squared_error",  
        "max_depth": 10,                # Profundidad máxima de los árboles
        "min_samples_split": 2,        # Mínimo de muestras para dividir un nodo
        "random_state": 42,
        "n_jobs": -1                   # Usa todos los núcleos
    }

    est = RandomForestRegressor(**params)
    pipe = Pipeline([("prep", pre), ("est", est)])

    # Entrena el modelo
    pipe.fit(X_train, y_train)

    # Predicciones
    y_pred = pipe.predict(X_test)

    # Calcula métricas
    metrics = calculate_metrics(y_test, y_pred)

    # Muestra resultados por consola
    print(f"[RandomForest] RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, R2={metrics['R2']:.3f}")

    # Registra en MLflow
    log_run_to_mlflow(exp, params=params, metrics=metrics, model=pipe) 
'''
'''
def run_cubist_regression(X_train, y_train, X_test, y_test, pre, cubist_params):
    exp = "CubistRegression"

    n_committees_list = cubist_params.get("n_committees", [1, 5, 10, 20])
    n_rules_list = cubist_params.get("n_rules", [10, 50, 100])

    for n_committees in n_committees_list:
        for n_rules in n_rules_list:
            params = {
                "n_committees": n_committees,
                "n_rules": n_rules,
            }

            est = Cubist(**params)
            pipe = Pipeline([("prep", pre), ("est", est)])

            # Entrena
            pipe.fit(X_train, y_train)

            # Predice
            y_pred = pipe.predict(X_test)

            # Métricas
            metrics = calculate_metrics(y_test, y_pred)
            print(f"[Cubist] {params} -> RMSE={metrics['RMSE']:.3f} MAE={metrics['MAE']:.3f} R2={metrics['R2']:.3f}")
            log_run_to_mlflow(exp, params=params, metrics=metrics, model=None)
'''

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
                
                # Train model (this will create child experiments)
                best_model, best_metrics = model_trainer.train_model(
                    model_config=model_config,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    preprocessor=pre
                )
                
                # Store results
                all_results[model_name] = {
                    'best_model': best_model,
                    'best_metrics': best_metrics,
                    'config': model_config
                }
                
                print(f"✓ {model_name} completed - Best RMSE: {best_metrics['RMSE']:.3f}")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                all_results[model_name] = None
        
        # 6 Log summary results in parent experiment (INSIDE the parent run context)
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        successful_models = {k: v for k, v in all_results.items() if v is not None}
        
        if successful_models:
            # Find best models for each criterion
            model_metrics = {name: result['best_metrics'] for name, result in successful_models.items()}
            
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
            
            # Log individual model results
            for model_name, result in successful_models.items():
                mlflow.log_metric(f"{model_name}_RMSE", result['best_metrics']['RMSE'])
                mlflow.log_metric(f"{model_name}_MAE", result['best_metrics']['MAE'])
                mlflow.log_metric(f"{model_name}_R2", result['best_metrics']['R2'])
                mlflow.log_metric(f"{model_name}_CV", result['best_metrics']['CV'])
        
        print(f"\nTotal models trained: {len(successful_models)}/{len(models_to_run)}")
        print(f"{'='*60}")
    
    return all_results



if __name__ == "__main__":
    main()
