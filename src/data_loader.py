import pandas as pd

def load_data(file_path="data/auto-mpg.data"):
    """
    Load the auto-mpg data file from the given file path and return a pandas DataFrame.
    Default file path is 'data/auto-mpg.data'.
    """
    # Define the column names as per the dataset specification
    column_names = [
        "mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", 
        "model_year", "origin", "car_name"
    ]
    
    # Load the data into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)
    
    # Handle missing values in the 'horsepower' column (e.g., replacing with the median or dropping rows)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna(subset=['horsepower'])  # Remove rows where 'horsepower' is missing
    
    return df
