import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    """Class to evaluate the performance of saved models on a test dataset."""

    def __init__(self, data_path: str, model_paths: dict):
        """
        Initialize the evaluator with the test data path and paths to the models.
        
        Args:
            data_path (str): Path to the processed test data.
            model_paths (dict): Dictionary containing model names as keys and paths as values.
        """
        self.data_path = data_path
        self.model_paths = model_paths

    def load_data(self):
        """Load and prepare the test data."""
        print("\n[INFO] Loading test data...")
        df = pd.read_csv(self.data_path)
        
        # Extract target and features
        y_test = df['mpg']
        X_test = df.drop(columns=['mpg', 'car_name'], errors='ignore').select_dtypes(include=['float64', 'int64'])
        
        print("[INFO] Test data loaded successfully!")
        return X_test, y_test

    def load_model(self, model_path):
        """Load a model from the given path."""
        try:
            model = joblib.load(model_path)
            print(f"[INFO] Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model from {model_path}: {e}")
            return None

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate the model and return performance metrics."""
        print(f"\n[INFO] Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print results
        print(f"\n[RESULTS] {model_name} Performance:")
        print(f"    - Mean Squared Error (MSE): {mse:.4f}")
        print(f"    - Mean Absolute Error (MAE): {mae:.4f}")
        print(f"    - R-Squared (R²): {r2:.4f}")
        
        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def run_evaluation(self):
        """Run evaluation on all models specified in model_paths."""
        # Load test data
        X_test, y_test = self.load_data()
        
        # Iterate through each model and evaluate
        results = {}
        for model_name, model_path in self.model_paths.items():
            model = self.load_model(model_path)
            if model:
                results[model_name] = self.evaluate_model(model, X_test, y_test, model_name)
        
        return results


if __name__ == "__main__":
    # Define the path to the test data and models
    processed_data_path = r'D:\Work\Study\Antern\MLops\Auto-MPG\data\processed\feature_engineered_data.csv'
    model_paths = {
        "Random Forest": "models/random_forest_tuned.joblib",
        "XGBoost": "models/xgboost_tuned.joblib"
    }
    
    # Initialize the ModelEvaluator with data and model paths
    evaluator = ModelEvaluator(processed_data_path, model_paths)
    
    # Run evaluation
    evaluation_results = evaluator.run_evaluation()
    
    # Display the final results
    print("\n[SUMMARY] Evaluation Results for All Models:")
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name} Metrics:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
