import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class ModelBuilder:
    """A class to build, train, and evaluate XGBoost and Random Forest models."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.model_rf = None
        self.model_xgb = None

    def preprocess_data(self):
        """Prepare features and target variable from the dataset."""
        # Assume 'mpg' is the target variable
        self.y = self.df['mpg']
        
        # Drop target and any non-numeric or irrelevant columns
        self.X = self.df.drop(columns=['mpg', 'car_name'], errors='ignore')
        
        # Ensure only numeric columns are used as features
        self.X = self.X.select_dtypes(include=['float64', 'int64'])
        
        return self.X, self.y

    def split_data(self, test_size=0.2, random_state=42):
        """Split the dataset into training and test sets."""
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def build_random_forest(self, n_estimators=100, random_state=42):
        """Initialize a Random Forest model."""
        self.model_rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def build_xgboost(self, n_estimators=100, learning_rate=0.1, random_state=42):
        """Initialize an XGBoost model."""
        self.model_xgb = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def train_model(self, model, X_train, y_train):
        """Train the model on the training data."""
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model and return performance metrics."""
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, mae, r2

    def save_model(self, model, model_name: str):
        """Save the trained model to disk."""
        model_path = f'models/{model_name}.joblib'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"{model_name} saved to {model_path}")

    def run(self):
        """Run the full pipeline: data preprocessing, model training, and evaluation."""
        # Preprocess data and split
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = self.split_data()

        # Build, train, and evaluate Random Forest
        self.build_random_forest()
        self.model_rf = self.train_model(self.model_rf, X_train, y_train)
        rf_mse, rf_mae, rf_r2 = self.evaluate_model(self.model_rf, X_test, y_test)
        print(f"Random Forest - MSE: {round(rf_mse, 2)}, MAE: {round(rf_mae, 2)}, R^2: {round(rf_r2, 2)}")
        self.save_model(self.model_rf, "random_forest")

        # Build, train, and evaluate XGBoost
        self.build_xgboost()
        self.model_xgb = self.train_model(self.model_xgb, X_train, y_train)
        xgb_mse, xgb_mae, xgb_r2 = self.evaluate_model(self.model_xgb, X_test, y_test)
        print(f"XGBoost - MSE: {round(xgb_mse, 2)}, MAE: {round(xgb_mae, 2)}, R^2: {round(xgb_r2, 2)}")
        self.save_model(self.model_xgb, "xgboost")


if __name__ == "__main__":
    # Define the path to the processed data
    processed_data_path = r'D:\Work\Study\Antern\MLops\Auto-MPG\data\processed\feature_engineered_data.csv'
    
    # Initialize ModelBuilder with the data path
    model_builder = ModelBuilder(processed_data_path)
    
    # Run the pipeline
    model_builder.run()
