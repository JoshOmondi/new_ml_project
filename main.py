from src.preprocess import load_data, preprocess_data
from src.train import train_and_evaluate
import pandas as pd
import os

def main():
    # Step 1: Preprocess raw data if needed
    input_path = 'data/auto-mpg.data'
    output_path = 'data/auto_mpg_clean.csv'

    if not os.path.exists(output_path):
        print("🔄 Cleaned data not found. Preprocessing raw data...")
        df = load_data(input_path)
        df_clean = preprocess_data(df)
        df_clean.to_csv(output_path, index=False)
        print(f"✅ Cleaned data saved to {output_path}")
    else:
        print("✅ Cleaned data already exists. Skipping preprocessing.")

    # Step 2: Train and evaluate the model
    model, mse, y_test, y_pred = train_and_evaluate()

    # Step 3: Output results
    print(f"Model Mean Squared Error (MSE): {mse}")

if __name__ == "__main__":
    main()
