from data_loader import load_data
from preprocess import preprocess_data
from train import train_and_evaluate
from visualize import plot_predictions


def main():
    df = load_data("data/auto-mpg.data")
    print("Original data shape:", df.shape)
    
    df_clean = preprocess_data(df)
    print("After cleaning:", df_clean.shape)  # Add this line
    
    model, y_test, y_pred = train_and_evaluate(df_clean)
    plot_predictions(y_test, y_pred)
