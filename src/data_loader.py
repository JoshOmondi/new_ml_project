import pandas as pd

def load_data(filepath):
    column_names = [
        "mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name"
    ]
    return pd.read_csv(filepath, delim_whitespace=True, names=column_names, na_values="?")
