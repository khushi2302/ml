import numpy as np
import unittest
from sklearn.linear_model import LinearRegression

# Assuming height and weight are numpy arrays
height = np.array([[1.75], [1.8], [1.65], [1.9], [1.7]])  # Example heights in meters
weight = np.array([65, 80, 55, 85, 60])  # Example weights in kilograms

# Train a linear regression model
reg = LinearRegression()
reg.fit(height, weight)

class TestLinearRegressionModel(unittest.TestCase):
    def test_model_accuracy(self):
        # Test data (example)
        test_height = np.array([[1.72], [1.85], [1.6]])  # New heights
        test_weight = np.array([63, 78, 52])  # Corresponding weights

        # Calculate model score (R^2)
        score = reg.score(test_height, test_weight)

        # Check if the model's score is above a certain threshold
        self.assertGreater(score, 0.9)  # Example threshold

if __name__ == '__main__':
    unittest.main()
