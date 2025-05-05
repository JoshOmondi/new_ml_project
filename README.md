# 🚗 Auto MPG Prediction Project

This machine learning project predicts car fuel efficiency (miles per gallon - MPG) based on various features such as engine size, horsepower, weight, and origin. The dataset is the classic [Auto MPG dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg) from the UCI Machine Learning Repository.

---

## 📂 Project Structure


---

## ⚙️ How It Works

1. **Preprocessing** (`preprocess.py`)
   - Loads raw dataset with missing values and inconsistent formatting
   - Cleans the data and performs one-hot encoding for categorical variables
   - Saves a cleaned version as `auto_mpg_clean.csv`

2. **Training & Evaluation** (`train.py`)
   - Splits data into training and testing sets
   - Trains a Linear Regression model
   - Evaluates using Mean Squared Error (MSE)

3. **Main Script** (`main.py`)
   - Orchestrates the full pipeline:
     - Loads and preprocesses the data (if not already done)
     - Trains the model and prints performance metrics

---

## 📦 Requirements

Make sure you have Python 3.7+ and install the dependencies below:

```bash
pip install pandas scikit-learn matplotlib
