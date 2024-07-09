import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            test_size = params['make_dataset']['test_size']
            return test_size
    except FileNotFoundError:
        print(f"Error: File '{params_path}' not found.")
        return 0.25  # Default test_size assuming params.yaml is missing or incorrect
    except KeyError:
        print(f"Error: 'test_size' parameter not found in '{params_path}'.")
        return 0.25  # Default test_size assuming test_size is not defined in params.yaml

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except FileNotFoundError:
        print(f"Error: File '{url}' not found.")
        return pd.DataFrame()  # Return an empty DataFrame on error

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(['id'], axis=1, inplace=True)
        return df
    except KeyError:
        print("Error: 'id' column not found.")
        return df  # Return unchanged DataFrame on error

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)  # Ensure directory exists or create it

        train_data.to_csv(os.path.join(data_path, "training_data.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "testing_data.csv"), index=False)
    except OSError as e:
        print(f"Error: Unable to create directory '{data_path}': {e}")

def main():
    params_path = 'params.yaml'
    test_size = load_params(params_path)

    raw_data_url = 'C:/Users/Srijan-DS/Documents/Projects/identify-profit-customer-profile/data/raw/raw.csv'
    df = read_data(raw_data_url)

    if not df.empty:
        final_df = process_data(df)

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        data_path = os.path.join("data", "interim")

        save_data(data_path, train_data, test_data)

        print('data saved')

if __name__ == '__main__':
    main()

