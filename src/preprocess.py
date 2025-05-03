import pandas as pd

def preprocess_data(df):
    df = df[df['horsepower'].notna()]  # Only drop rows with missing horsepower
    df = df.drop("car_name", axis=1)
    df = pd.get_dummies(df, columns=["origin"], prefix="origin")
    return df
