import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title("Actual vs Predicted MPG")
    plt.grid(True)
    plt.show()
