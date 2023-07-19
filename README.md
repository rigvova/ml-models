# Machine Learning Models

This repository contains classical machine learning models implemented in Python, including Linear Regression,
Logistic Regression, and Decision Trees. The models are designed to be simple, easy-to-understand, and effective.

## Repository Structure
The repository consists of the following python scripts:

* `linear_regression.py`
* `logistic_regression.py`
* `tree_regressor.py`

## Usage
To use the LinearRegression class, create an instance of the class, then use the fit method to train on your data. 
Use the predict method to make predictions. Example:

```
import numpy as np
from linear_regression import LinearRegression

# Generate some simple linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.flatten() + np.random.normal(scale=1, size=100)

# Initialize the Linear Regression model
model = LinearRegression(lr=0.01, alpha=0.0, beta=0.0)

# Train the model
model.fit(X, y)

# Predict on new data
X_new = np.array([[5.5], [6.5], [7.5]])
predictions = model.predict(X_new)

print()
print("True y_test:", 3 * X_new.flatten())
print("Predictions:", predictions)
```

---
Please refer to the individual Python scripts for detailed information about each algorithm implementation.