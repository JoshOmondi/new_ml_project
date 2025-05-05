import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_and_evaluate():
    # Load and clean data
    df = pd.read_csv('data/auto_mpg_clean.csv')
    
    # Drop non-numeric column
    df = df.drop(columns=['car name'])

    # Features and target
    X = df.drop('mpg', axis=1)
    y = df['mpg']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse, y_test, y_pred
