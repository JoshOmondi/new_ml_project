
import pandas as pd

def load_data(file_path):
    """Loads the raw dataset from the specified file path."""
    columns = [
        "mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration",
        "model year", "origin", "car name"
    ]
    # Load the dataset with space-delimited format
    df = pd.read_csv(file_path, header=None, names=columns, delimiter=r'\s+', na_values="?")
    return df

def preprocess_data(df):
    """Preprocesses the dataset: handles missing values and encodes categorical variables."""
    # Fill missing 'horsepower' values with the median
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

    # Convert 'origin' column into dummy variables (one-hot encoding)
    df = pd.get_dummies(df, columns=['origin'], drop_first=True)
    
    return df
