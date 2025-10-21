import pandas as pd
import sys
import yaml
import os
import numpy as np


class DataLoader:
    """Class to load data from different sources."""
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.data = None
    
    def load_data(self, input_path):
        """
        Load data from a CSV file.
        
        Args:
            input_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File {input_path} does not exist")
            
            self.data = pd.read_csv(input_path)
            
            if self.data.empty:
                raise pd.errors.EmptyDataError(f"File {input_path} is empty")
            
            print(f"Data loaded successfully from {input_path}")
            print(f"Data shape: {self.data.shape}")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise


class Preprocessor:
    """Class to preprocess steel energy data."""
    
    def __init__(self):
        """Initialize the Preprocessor."""
        self.numeric_cols = [
            'Usage_kWh', 
            'Lagging_Current_Reactive.Power_kVarh', 
            'Leading_Current_Reactive_Power_kVarh', 
            'CO2(tCO2)', 
            'Lagging_Current_Power_Factor', 
            'Leading_Current_Power_Factor', 
            'NSM'
        ]
        
        self.weekday_mapping = {
            'wEEKDAY': 'Weekday',
            'wEEKEND': 'Weekend',
            'NAN': np.nan
        }
        
        self.day_of_week_mapping = {
            'mONDAY': 'Monday',
            'tUESDAY': 'Tuesday',
            'wEDNESDAY': 'Wednesday',
            'tHURSDAY': 'Thursday',
            'fRIDAY': 'Friday',
            'sATURDAY': 'Saturday',
            'sUNDAY': 'Sunday',
            'NAN': np.nan
        }
        
        self.load_type_mapping = {
            'lIGHT_lOAD': 'Light_Load',
            'mEDIUM_lOAD': 'Medium_Load',
            'mAXIMUM_lOAD': 'Maximum_Load',
            'NAN': np.nan
        }
    
    def _drop_outliers_iqr(self, df, col, k=1.5, return_mask=False):
        """Remove outliers from `col` using global IQR (without grouping)."""
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k*iqr, q3 + k*iqr
        mask = df[col].isna() | df[col].between(lo, hi)
        if return_mask:
            return mask, (lo, hi)
        return df[mask].copy(), (lo, hi)
    
    def preprocess(self, df, output_path=None):
        """
        Preprocess data by applying all necessary transformations.
        
        Args:
            df (pd.DataFrame): DataFrame with data to preprocess
            output_path (str, optional): Path where to save preprocessed data
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df_processed = df.copy()
        
        # Clean numerical variables
        df_processed = self._clean_numerical_columns(df_processed)
        
        # Clean categorical variables
        df_processed = self._clean_categorical_columns(df_processed)
        
        # Remove rows with NaN
        df_processed = df_processed.dropna()
        
        # Save if output_path is specified
        if output_path:
            self._save_processed_data(df_processed, output_path)
        
        return df_processed
    
    def _clean_numerical_columns(self, df):
        """Clean numerical columns."""
        df_clean = df.copy()
        
        # Convert 'date' column to datetime
        df_clean['date'] = pd.to_datetime(df['date'].str.strip(), format='%d/%m/%Y %H:%M', errors='raise')
        
        # Remove mixed_type_col column
        df_clean.drop(columns=['mixed_type_col'], inplace=True)
        
        # Clean numerical columns
        for col in self.numeric_cols:
            df_clean[col] = pd.to_numeric(
                df_clean[col].replace([
                    'invalid', '?', 'error', ' NAN ', ' ? ', ' null ', 
                    ' ERROR ', ' n/a ', ' INVALID '
                ], np.nan), 
                errors='raise'
            )
        
        # Convert 'NSM' column to Int64
        df_clean['NSM'] = df_clean['NSM'].astype('Int64')
        
        # Remove outliers with different k values
        outlier_configs = [
            ('Usage_kWh', 3.0),
            ('Lagging_Current_Reactive.Power_kVarh', 3.0),
            ('Leading_Current_Reactive_Power_kVarh', 20),
            ('CO2(tCO2)', 3),
            ('Lagging_Current_Power_Factor', 3),
            ('Leading_Current_Power_Factor', 300)
        ]
        
        for col, k in outlier_configs:
            df_clean, _ = self._drop_outliers_iqr(df_clean, col, k=k)
        
        # Filter NSM <= 86399 (seconds in a day)
        df_clean = df_clean[df_clean['NSM'] <= 86399].copy()
        
        return df_clean
    
    def _clean_categorical_columns(self, df):
        """Clean categorical columns."""
        df_clean = df.copy()
        
        # Clean WeekStatus
        df_clean['WeekStatus'] = df['WeekStatus'].str.strip().replace(self.weekday_mapping)
        
        # Clean Day_of_week
        df_clean['Day_of_week'] = df['Day_of_week'].str.strip().replace(self.day_of_week_mapping)
        
        # Clean Load_Type
        df_clean['Load_Type'] = df['Load_Type'].str.strip().replace(self.load_type_mapping)
        
        return df_clean
    
    def _save_processed_data(self, df, output_path):
        """Save preprocessed data."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, header=True, index=False)
        print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    # Load parameters from params.yaml
    params = yaml.safe_load(open("params.yaml"))['preprocess']

    # Use the new classes
    data_loader = DataLoader()
    preprocessor = Preprocessor()
    
    # Load data
    df = data_loader.load_data(params["input"])
    
    # Preprocess data
    df_processed = preprocessor.preprocess(df, params["output"])
    
    print(f"Preprocessing completed. Final shape: {df_processed.shape}")
