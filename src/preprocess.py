import pandas as pd
import sys
import yaml
import os
import numpy as np

## Load parameters from param.yaml

params=yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path,output_path):
    #Reads the raw data from the input path
    df=pd.read_csv(input_path)
    df2 = df.copy()
    
    numeric_cols = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']

    for col in numeric_cols:
        df2[col] = pd.to_numeric(df2[col].replace(['invalid', '?', 'error', ' NAN ', ' ? ', ' null ', ' ERROR ', ' n/a ', ' INVALID '], np.nan), errors='coerce')


    df_final = df2.copy()
    #Saves the preprocessed data to the output path
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df_final.to_csv(output_path,header=None,index=False)
    print(f"Preprocesses data saved to {output_path}")

if __name__=="__main__":
    preprocess(params["input"],params["output"])
