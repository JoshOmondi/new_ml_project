import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.preprocess import load_data, preprocess_data

def train_and_evaluate():
    # Load and preprocess data
    df = load_data("data/auto-mpg.data-original")
    df_clean = preprocess_data(df)
    
    # Handle missing values in the target variable 'mpg'
    print(f"Missing values in target (mpg): {df_clean['mpg'].isnull().sum()}")

    # Drop rows where 'mpg' is NaN
    df_clean = df_clean.dropna(subset=['mpg'])
    
    # Separate features and target variable
    X = df_clean.drop(columns=['mpg', 'car name'])  # Features
    y = df_clean['mpg']  # Target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and calculate MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse, y_test, y_pred

if __name__ == "__main__":
    model, mse, y_test, y_pred = train_and_evaluate()
    print(f"Model MSE: {mse}")
