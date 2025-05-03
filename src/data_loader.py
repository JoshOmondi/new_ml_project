import pandas as pd
import os

def load_data():
    """
    Loads the auto-mpg dataset from the 'data/auto-mpg.data-original' file.
    """
    # Get the current directory (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the data file
    file_path = os.path.join(project_root, '..', 'data', 'auto-mpg.data-original')
    
    # Print the file path for debugging
    print(f"Looking for file at: {file_path}")
    
    # Ensure the file exists before trying to read
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    # Define column names for the dataset
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # Load the dataset
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=column_names)

    return df
