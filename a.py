"""
Simple Linear Regression Example
A basic ML trial demonstrating linear regression with scikit-learn
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Constants for data generation
NUM_SAMPLES = 100
TRUE_SLOPE = 2.5
TRUE_INTERCEPT = 1.5
NOISE_SCALE = 2


def main():
    # Generate sample data
    np.random.seed(42)
    x = np.random.rand(NUM_SAMPLES, 1) * 10  # NUM_SAMPLES samples, 1 feature
    y = TRUE_SLOPE * x.squeeze() + TRUE_INTERCEPT + np.random.randn(NUM_SAMPLES) * NOISE_SCALE  # Linear relationship with noise
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=" * 50)
    print("Linear Regression Model Results")
    print("=" * 50)
    print(f"Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 50)
    
    # Plot the results
    # Sort data for smooth line visualization
    sort_idx = x_test.squeeze().argsort()
    x_test_sorted = x_test[sort_idx]
    y_test_sorted = y_test[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test_sorted, y_test_sorted, color='blue', alpha=0.6, label='Actual data')
    plt.plot(x_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Predicted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_plot.png')
    plt.close()  # Free up memory
    print("Plot saved as 'linear_regression_plot.png'")


if __name__ == "__main__":
    main()
