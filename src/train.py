import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

def load_features():
    df = pd.read_csv('data/auto_mpg_clean.csv')
    df = df.drop(columns=['car name'])
    X = df.drop('mpg', axis=1)
    y = df['mpg']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue', edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title(f"{model_name} — Predicted vs Actual MPG")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_features()

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    best_model = None
    best_mse = float('inf')
    best_name = ""
    best_pred = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append((name, mse, r2))

        if mse < best_mse:
            best_model = model
            best_mse = mse
            best_name = name
            best_pred = y_pred

    # Print performance summary
    print("\n📊 Model Performance Summary:")
    print(f"{'Model':<20} {'MSE':<10} {'R² Score':<10}")
    print("-" * 42)
    for name, mse, r2 in results:
        print(f"{name:<20} {mse:<10.3f} {r2:<10.3f}")

    # Visualize predictions for best model
    print(f"\n🧠 Best Model: {best_name}")
    plot_predictions(y_test, best_pred, best_name)

    return best_model, best_mse, y_test, best_pred
