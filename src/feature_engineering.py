import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

class AutoMPGFeatureEngineering:
    """Performs feature engineering for the Auto MPG dataset."""

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by filling them with median or mean for numerical columns."""
        # Fill numerical columns with their median
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)

        # For categorical columns, we can fill with the most frequent value
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df

    def ensure_numeric_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Ensure that specified columns are numeric, converting errors to NaN."""
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between columns."""
        # Ensure horsepower and weight are numeric
        df = self.ensure_numeric_columns(df, ['horsepower', 'weight'])

        # Example: Create interaction between horsepower and weight
        df['horsepower_weight_interaction'] = df['horsepower'] * df['weight']
        
        # You could also try more complex interactions (e.g., ratios or differences)
        df['horsepower_weight_ratio'] = df['horsepower'] / (df['weight'] + 1e-5)  # avoid division by zero
        
        return df

    def log_transform_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Apply log transformation to reduce skewness."""
        for col in columns:
            df[col] = df[col].apply(lambda x: np.log(x + 1) if x > 0 else 0)
        return df

    def binning(self, df: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
        """Convert a continuous variable into discrete bins."""
        df[f'{column}_bin'] = pd.cut(df[column], bins=bins, labels=False, include_lowest=True)
        return df

    def scale_features(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Scale numerical features to have mean=0 and variance=1 (standardization)."""
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features (degree 2 for quadratic features)."""
        poly = PolynomialFeatures(degree)
        poly_features = poly.fit_transform(df[['horsepower', 'weight']])  # Use horsepower and weight as an example
        
        # Create new columns for the polynomial features
        poly_feature_names = poly.get_feature_names_out(['horsepower', 'weight'])
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
        
        # Concatenate the new features with the original dataframe
        df = pd.concat([df, df_poly], axis=1)
        return df

    def clean_car_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize car names."""
        # Extract the first word from car name (similar to `str_split` in R)
        df['car_name'] = df['car_name'].str.split().str[0]

        # Standardize car names by replacing incorrect values
        df['car_name'] = df['car_name'].replace({
            'chevroelt': 'chevrolet',
            'hi': np.nan,
            'maxda': 'mazda',
            'mercedes-benz': 'mercedes',
            'toyouta': 'toyota',
            'vokswagen': 'volkswagen'
        })

        # Convert 'car_name' to categorical
        df['car_name'] = df['car_name'].astype('category')
        
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Clean and standardize the 'car_name'
        df = self.clean_car_name(df)
        
        # Step 3: Create interaction features (e.g., horsepower * weight)
        df = self.create_interaction_features(df)
        
        # Step 4: Apply log transformation to skewed features (e.g., horsepower, weight)
        df = self.log_transform_features(df, ['horsepower', 'weight'])
        
        # Step 5: Apply binning to horsepower and weight (optional)
        df = self.binning(df, 'horsepower', bins=5)
        df = self.binning(df, 'weight', bins=5)
        
        # Step 6: Standardize numerical features
        df = self.scale_features(df, ['mpg', 'horsepower', 'weight'])
        
        # Step 7: Create polynomial features (optional)
        df = self.create_polynomial_features(df)
        
        return df


def process_auto_mpg(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the Auto MPG dataset."""
    feature_engineer = AutoMPGFeatureEngineering()
    
    # Apply feature engineering transformations
    df_engineered = feature_engineer.engineer_features(df)
    
    return df_engineered


if __name__ == "__main__":
    from ingest_data import DataIngestor

    # Connect to PostgreSQL and load data from the 'auto_mpg' table
    DB_URL = "postgresql://postgres:*963.*963.Rutu@localhost:5432/super_30"
    ingestor = DataIngestor(DB_URL)
    df = ingestor.load_data('auto_mpg')  # Load the dataset

    # Process the ingested data (apply feature engineering)
    df_engineered = process_auto_mpg(df)

    # Display the first few rows of the engineered DataFrame
    print(df_engineered.head())

    # Define the output path
    output_path = r'D:\Work\Study\Antern\MLops\Auto-MPG\data\processed\feature_engineered_data.csv'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Save the engineered DataFrame to a CSV file
        df_engineered.to_csv(output_path, index=False)
        print(f"Feature-engineered data saved to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
