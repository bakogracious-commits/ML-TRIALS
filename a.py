"""
Simple Linear Regression Example
A basic ML trial demonstrating linear regression with scikit-learn
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def main():
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2.5 * X.squeeze() + 1.5 + np.random.randn(100) * 2  # Linear relationship with noise
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
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
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_plot.png')
    print("Plot saved as 'linear_regression_plot.png'")


if __name__ == "__main__":
    main()
