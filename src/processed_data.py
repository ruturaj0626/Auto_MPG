import pandas as pd
import os

class AutoMPGProcessor:
    """Processes the Auto MPG dataset with outlier handling."""

    def handle_outliers(self, df: pd.DataFrame, columns: list, multiplier: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numerical columns using the IQR method."""
        for column in columns:
            # Ensure the column is numeric and handle non-numeric values
            df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')

            # Check how many NaNs were created
            nan_count = df[column].isna().sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} NaNs created in column {column} after conversion.")

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            initial_shape = df.shape
            # Filter the DataFrame to exclude rows with outliers
            df = df[(df[column] >= Q1 - multiplier * IQR) & (df[column] <= Q3 + multiplier * IQR)]
            final_shape = df.shape
            print(f"Removed {initial_shape[0] - final_shape[0]} rows due to outliers in column {column}")
        
        return df

def process_auto_mpg(df: pd.DataFrame) -> pd.DataFrame:
    """Process the Auto MPG DataFrame by outlier removal."""
    processor = AutoMPGProcessor()

    # Step 2: Handle outliers in 'mpg', 'horsepower', and 'weight'
    df_cleaned = processor.handle_outliers(df, ['mpg', 'horsepower', 'weight'])

    return df_cleaned

if __name__ == "__main__":
    from ingest_data import DataIngestor

    # Connect to PostgreSQL and load data from the 'auto_mpg' table
    DB_URL = "postgresql://postgres:*963.*963.Rutu@localhost:5432/super_30"
    ingestor = DataIngestor(DB_URL)
    df = ingestor.load_data('auto_mpg')  # Load the dataset

    # Process the ingested data (only outlier handling)
    df_cleaned = process_auto_mpg(df)

    # Display the first few rows of the cleaned DataFrame
    print(df_cleaned.head())

    # Define the output path
    output_path = r'D:\Work\Study\Antern\MLops\Auto-MPG\data\processed\processed_data.csv'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Save the cleaned DataFrame to a CSV file
        df_cleaned.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
