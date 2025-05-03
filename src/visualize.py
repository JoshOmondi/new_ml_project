import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions_vs_true(y_test, y_pred):
    """
    Plot the true values vs. predicted values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolors='black', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title("True Values vs. Predicted Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()

def plot_residuals(y_test, y_pred):
    """
    Plot the residuals (True - Predicted).
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals (True - Predicted)")
    plt.ylabel("Frequency")
    plt.show()

def plot_feature_distribution(df):
    """
    Plot the distribution of continuous features in the dataset.
    """
    continuous_features = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
    df[continuous_features].hist(bins=30, figsize=(12, 8))
    plt.suptitle("Feature Distributions")
    plt.show()
