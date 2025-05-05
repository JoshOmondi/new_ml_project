from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

def load_features():
    df = pd.read_csv('data/auto_mpg_clean.csv')
    df = df.drop(columns=['car name'])
    X = df.drop('mpg', axis=1)
    y = df['mpg']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_features()

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return model, mse, y_test, y_pred
