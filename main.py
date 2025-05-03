from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_and_evaluate
from src.visualize import plot_predictions_vs_true, plot_residuals, plot_feature_distribution

def main():
    # Load the data
    df = load_data()  # This will load data from 'data/auto-mpg.data'
    
    # Preprocess the data
    df_clean = preprocess_data(df)  # Clean the data using preprocessing function
    
    # Train the model and evaluate
    model, mse, y_test, y_pred = train_and_evaluate(df_clean)  # Train the model
    
    # Print evaluation results
    print(f"Model: Linear Regression")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"True values (y_test): {y_test.head()}")
    print(f"Predicted values (y_pred): {y_pred[:5]}")
    
    # Visualize results
    plot_predictions_vs_true(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_feature_distribution(df_clean)

if __name__ == "__main__":
    main()
