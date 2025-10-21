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

# Model configurations
MODEL_CONFIGS = {
    "LinearRegression": {
        "name": "LinearRegression",
        "estimator": LinearRegression,
        "params": [{"fit_intercept": True}, {"fit_intercept": False}],
        "experiment_name": "LinearRegression"
    },
    "KNN": {
        "name": "KNNRegression",
        "estimator": KNeighborsRegressor,
        "params": [
            {"n_neighbors": 3, "weights": "uniform", "p": 1},
            {"n_neighbors": 5, "weights": "distance", "p": 2},
            {"n_neighbors": 7, "weights": "uniform", "p": 2},
            {"n_neighbors": 11, "weights": "distance", "p": 1}
        ],
        "experiment_name": "KNNRegression"
    },
    "CART": {
        "name": "CARTRegression",
        "estimator": DecisionTreeRegressor,
        "params": [
            {"ccp_alpha": 0.0, "random_state": 42},
            {"ccp_alpha": 0.1, "random_state": 42},
            {"ccp_alpha": 0.2, "random_state": 42},
            {"ccp_alpha": 0.3, "random_state": 42},
            {"ccp_alpha": 0.4, "random_state": 42},
            {"ccp_alpha": 0.5, "random_state": 42}
        ],
        "experiment_name": "CARTRegression"
    },
    "RandomForest": {
        "name": "RandomForestRegression",
        "estimator": RandomForestRegressor,
        "params": [{
            "n_estimators": 100,
            "criterion": "squared_error",
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1
        }],
        "experiment_name": "RandomForestRegression"
    },
    "Cubist": {
        "name": "CubistRegression",
        "estimator": Cubist,
        "params": [
            {"n_committees": 1, "n_rules": 10},
            {"n_committees": 5, "n_rules": 50},
            {"n_committees": 10, "n_rules": 100},
            {"n_committees": 20, "n_rules": 10}
        ],
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

def load_params(path="params.yaml"):
    if Path(path).exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError("params.yaml no encontrado")

def log_run_to_mlflow(experiment_name, params, metrics, model=None, run_name=None):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        #if model is not None:
            #mlflow.sklearn.log_model(model, "model")

def calculate_metrics(y_true, y_pred):
    """
    Calculate RMSE, MAE, and R2 metrics for regression models.
    Also prints the metrics to console.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing RMSE, MAE, and R2 metrics
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cv = cv_percent(y_true, y_pred)
    
    # Print metrics
    #print('RMSE:', rmse)
    #print('MAE:', mae)
    #print('R2:', r2)
    #print('CV', cv)
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "CV": cv}

def cv_percent(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    return 100.0 * rmse / np.mean(y_true)

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
        self.metrics_calculator = metrics_calculator
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
        if self.metrics_calculator:
            metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred)
        else:
            metrics = calculate_metrics(y_test, y_pred)  # Fallback to function
        
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
    
    def get_model_config(self, model_name):
        """
        Get model configuration by name.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS")
        
        return MODEL_CONFIGS[model_name]



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

def hyperparameter_tuning(X_train, y_train, pre, est, param_grid):
    pipe = Pipeline([("prep", pre), ("est", est)])
    grid_search=GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2,
                             scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    return grid_search


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


def main():
    """Main function that runs all regression models in a parent experiment."""

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    params = load_params()
    data_path = params["preprocess"]["output"]
    split_cfg = params.get("split", {})
    train_cfg = params.get("train", {})
    test_size = split_cfg.get("test_size", 0.2)
    
    # Get list of models to run (default: all models)
    models_to_run = train_cfg.get("models_to_run", list(MODEL_CONFIGS.keys()))
    
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
    model_trainer = ModelTrainer()
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
                model_config = model_trainer.get_model_config(model_name)
                
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
            # Find overall best model
            best_overall_model = min(successful_models.items(), 
                                   key=lambda x: x[1]['best_metrics']['RMSE'])
            
            best_model_name, best_model_data = best_overall_model
            
            print(f"Best overall model: {best_model_name}")
            print(f"Best RMSE: {best_model_data['best_metrics']['RMSE']:.3f}")
            print(f"Best MAE: {best_model_data['best_metrics']['MAE']:.3f}")
            print(f"Best R2: {best_model_data['best_metrics']['R2']:.3f}")
            
            # Log best model metrics in parent experiment
            mlflow.log_metric("best_overall_RMSE", best_model_data['best_metrics']['RMSE'])
            mlflow.log_metric("best_overall_MAE", best_model_data['best_metrics']['MAE'])
            mlflow.log_metric("best_overall_R2", best_model_data['best_metrics']['R2'])
            mlflow.log_param("best_overall_model", best_model_name)
            
            # Log individual model results
            for model_name, result in successful_models.items():
                mlflow.log_metric(f"{model_name}_RMSE", result['best_metrics']['RMSE'])
                mlflow.log_metric(f"{model_name}_MAE", result['best_metrics']['MAE'])
                mlflow.log_metric(f"{model_name}_R2", result['best_metrics']['R2'])
        
        print(f"\nTotal models trained: {len(successful_models)}/{len(models_to_run)}")
        print(f"{'='*60}")
    
    return all_results



if __name__ == "__main__":
    main()
