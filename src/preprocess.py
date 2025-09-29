import pandas as pd
import sys
import yaml
import os
import numpy as np

## Load parameters from param.yaml

params=yaml.safe_load(open("params.yaml"))['preprocess']


def drop_outliers_iqr(df, col, k=1.5, return_mask=False):
    """Elimina outliers de `col` con IQR global (sin agrupar)."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    mask = df[col].isna() | df[col].between(lo, hi)
    if return_mask:
        return mask, (lo, hi)
    return df[mask].copy(), (lo, hi)

def preprocess(input_path,output_path):
    #Reads the raw data from the input path
    df=pd.read_csv(input_path)
    df2 = df.copy()

    ####### Numerical variables cleaning ########

    # Convert 'date' column to datetime
    df2['date'] = pd.to_datetime(df['date'].str.strip(), format='%d/%m/%Y %H:%M', errors='raise')

    numeric_cols = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']

    #Drop the mixed_type_col column
    df2.drop(columns=['mixed_type_col'], inplace=True)

    for col in numeric_cols:
        df2[col] = pd.to_numeric(df2[col].replace(['invalid', '?', 'error', ' NAN ', ' ? ', ' null ', ' ERROR ', ' n/a ', ' INVALID '], np.nan), errors='coerce')

    # Convert 'NSM' column to Int64
    df2['NSM'] = df2['NSM'].astype('Int64')

    df2_1, (_, _) = drop_outliers_iqr(df2, 'Usage_kWh', k=3.0)

    df2_2, (_, _) = drop_outliers_iqr(df2_1, 'Lagging_Current_Reactive.Power_kVarh', k=3.0)

    df2_3, (_, _) = drop_outliers_iqr(df2_2, 'Leading_Current_Reactive_Power_kVarh', k=20)

    df2_4, (_, _) = drop_outliers_iqr(df2_3, 'CO2(tCO2)', k=3)

    df2_5, (_, _) = drop_outliers_iqr(df2_4, 'Lagging_Current_Power_Factor', k=3)

    df2_6, (_, _) = drop_outliers_iqr(df2_5, 'Leading_Current_Power_Factor', k=300)

    #Drop the rows with NSM > 86399 because the NSM is the number of seconds in a day
    df2_7 = df2_6[df2_6['NSM'] <= 86399].copy()

    ####### Categorical variables cleaning ########
    df3 = df2_7.copy()

    weekday_mapping = {
        'wEEKDAY':'Weekday',
        'wEEKEND':'Weekend',
        'NAN' : np.nan
    }
    df3['WeekStatus'] = df2_7['WeekStatus'].str.strip().replace(weekday_mapping)

    day_of_week_mapping = {
        'mONDAY': 'Monday',
        'tUESDAY': 'Tuesday',
        'wEDNESDAY': 'Wednesday',
        'tHURSDAY': 'Thursday',
        'fRIDAY': 'Friday',
        'sATURDAY': 'Saturday',
        'sUNDAY': 'Sunday',
        'NAN' : np.nan
    }
    df3['Day_of_week'] = df2_7['Day_of_week'].str.strip().replace(day_of_week_mapping)

    load_type_mapping = {
        'lIGHT_lOAD': 'Light_Load',
        'mEDIUM_lOAD': 'Medium_Load',
        'mAXIMUM_lOAD': 'Maximum_Load',
        'NAN' : np.nan
    }
    df3['Load_Type'] = df2_7['Load_Type'].str.strip().replace(load_type_mapping)

    #Drop the rows with NaN values
    df4 = df3.dropna()

    df_final = df4.copy()
    #Saves the preprocessed data to the output path
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df_final.to_csv(output_path,header=None,index=False)
    print(f"Preprocesses data saved to {output_path}")

if __name__=="__main__":
    preprocess(params["input"],params["output"])
