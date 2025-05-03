
from src.train import train_and_evaluate

def main():
    # Train the model and evaluate
    model, mse, y_test, y_pred = train_and_evaluate()
    
    # Print the results
    print(f"Model Mean Squared Error (MSE): {mse}")
    
if __name__ == "__main__":
    main()
