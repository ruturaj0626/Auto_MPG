from sqlalchemy import create_engine
import pandas as pd

class DataIngestor:
    """Ingests data directly from a PostgreSQL database."""

    def __init__(self, db_uri: str):
        self.engine = create_engine(db_uri)

    def load_data(self, table_name: str) -> pd.DataFrame:
        """Load data from a specific table into a DataFrame."""
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, self.engine)
        return df

# Usage Example:
DB_URL = "postgresql://postgres:*963.*963.Rutu@localhost:5432/super_30"
ingestor = DataIngestor(DB_URL)
df = ingestor.load_data('auto_mpg')
print(df.head(2))
