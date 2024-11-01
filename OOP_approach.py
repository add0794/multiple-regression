import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        """Initialize the model with training and optional test data."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = LinearRegression()  # Instantiate the model
        self.trained = False

    def train(self):
        """Train the model on the training data."""
        self.model.fit(self.X_train, self.y_train)
        self.trained = True
        print("Model trained.")

    def predict(self, X):
        """Make predictions on new data."""
        if not self.trained:
            raise Exception("Model is not trained yet. Call train() before predicting.")
        return self.model.predict(X)

    def evaluate(self):
        """Evaluate the model on the test data, if available."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not provided.")
        predictions = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r_squared = r2_score(self.y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r_squared}")
        return mse, r_squared

    def plot_predictions(self):
        """Plot predictions vs actual values if test data is available."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not provided.")
        predictions = self.predict(self.X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, predictions, color='blue', alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.show()

    def summary(self):
        """Print model coefficients and intercept."""
        print("Model Coefficients:", pd.DataFrame(self.model.coef_, index=self.X_train.columns, columns=['Coefficient']))
        print("Intercept:", self.model.intercept_)

# Download and load data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = pd.DataFrame(
    np.hstack([raw_df.values[::2], raw_df.values[1::2]]),
    columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
)

# Split data into features and target
X = data.drop('MEDV', axis=1)
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Usage Example
# Assuming you already have split data: X_train, X_test, y_train, y_test

# Step 1: Instantiate the model
model = LinearRegressionModel(X_train, y_train, X_test, y_test)

# Step 2: Train the model
model.train()

# Step 3: Evaluate the model
model.evaluate()

# Step 4: View model summary (coefficients and intercept)
model.summary()

# Step 5: Plot predictions vs actual values
model.plot_predictions()
