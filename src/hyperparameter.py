import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np

class HyperparameterTuner:
    """A class to perform hyperparameter tuning for Random Forest and XGBoost models."""

    def __init__(self, data_path: str, random_search: bool = True, n_iter: int = 20):
        self.data_path = data_path
        self.random_search = random_search
        self.n_iter = n_iter  # Number of iterations for RandomizedSearchCV if random_search=True
        self.X = None
        self.y = None

    def preprocess_data(self):
        """Prepare features and target variable from the dataset."""
        print("\n[INFO] Loading and preprocessing data...")
        df = pd.read_csv(self.data_path)
        
        # Define target and features
        self.y = df['mpg']
        self.X = df.drop(columns=['mpg', 'car_name'], errors='ignore').select_dtypes(include=['float64', 'int64'])
        
        print("[INFO] Data loaded successfully!")
        return self.X, self.y

    def split_data(self, test_size=0.2, random_state=42):
        """Split the dataset into training and test sets."""
        print("\n[INFO] Splitting data into train and test sets...")
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def random_forest_hyperparameters(self):
        """Define the hyperparameters grid for Random Forest."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

    def xgboost_hyperparameters(self):
        """Define the hyperparameters grid for XGBoost."""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.3, 0.5]
        }

    def tune_model(self, model, param_grid, X_train, y_train, model_name):
        """Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV."""
        print(f"\n[INFO] Starting hyperparameter tuning for {model_name}...")
        search_type = "RandomizedSearchCV" if self.random_search else "GridSearchCV"
        print(f"[INFO] Using {search_type} with {self.n_iter if self.random_search else 'all'} combinations.")
        
        if self.random_search:
            search = RandomizedSearchCV(model, param_grid, n_iter=self.n_iter, cv=3, n_jobs=-1, verbose=1, random_state=42)
        else:
            search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
            
        search.fit(X_train, y_train)
        print(f"[INFO] Best parameters for {model_name}: {search.best_params_}\n")
        return search.best_estimator_

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate the tuned model and return performance metrics."""
        print(f"\n[INFO] Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n[RESULTS] {model_name} Performance:")
        print(f"    - Mean Squared Error (MSE): {mse:.4f}")
        print(f"    - Mean Absolute Error (MAE): {mae:.4f}")
        print(f"    - R-Squared (R²): {r2:.4f}")
        return mse, mae, r2

    def save_model(self, model, model_name: str):
        """Save the trained model to disk."""
        model_path = f'models/{model_name}.joblib'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"[INFO] {model_name} saved to {model_path}")

    def run(self):
        """Run the hyperparameter tuning and evaluation for both models."""
        # Preprocess data and split
        self.preprocess_data()
        X_train, X_test, y_train, y_test = self.split_data()

        # Hyperparameter tuning for Random Forest
        rf_model = RandomForestRegressor(random_state=42)
        rf_param_grid = self.random_forest_hyperparameters()
        best_rf = self.tune_model(rf_model, rf_param_grid, X_train, y_train, "Random Forest")
        self.evaluate_model(best_rf, X_test, y_test, "Random Forest")
        self.save_model(best_rf, "random_forest_tuned")

        # Hyperparameter tuning for XGBoost
        xgb_model = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
        xgb_param_grid = self.xgboost_hyperparameters()
        best_xgb = self.tune_model(xgb_model, xgb_param_grid, X_train, y_train, "XGBoost")
        self.evaluate_model(best_xgb, X_test, y_test, "XGBoost")
        self.save_model(best_xgb, "xgboost_tuned")


if __name__ == "__main__":
    # Define the path to the processed data
    processed_data_path = r'D:\Work\Study\Antern\MLops\Auto-MPG\data\processed\feature_engineered_data.csv'
    
    # Initialize HyperparameterTuner with the data path
    tuner = HyperparameterTuner(processed_data_path, random_search=True, n_iter=20)
    
    # Run the tuning pipeline
    tuner.run()
