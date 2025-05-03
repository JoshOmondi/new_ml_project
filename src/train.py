import pandas as pd

def preprocess_data(df):
    """
    Perform data preprocessing steps:
    - Convert 'origin' to categorical variable.
    - Normalize continuous features.
    """
    # Convert 'origin' to categorical (dummy encoding)
    df = pd.get_dummies(df, columns=['origin'], drop_first=True)
    
    # Normalize continuous features
    continuous_columns = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
    df[continuous_columns] = df[continuous_columns].apply(lambda x: (x - x.mean()) / x.std())
    
    return df
