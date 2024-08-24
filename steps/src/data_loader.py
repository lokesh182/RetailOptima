import pandas as pd
from sqlalchemy import create_engine , exc

class DataLoader:
    """class to load data from a sql database"""
    
    def __init__(self,db_uri: str):
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.data = None
        
    def load_data(self, table_name: str) -> pd.DataFrame:
        """Load data from a table in the database"""
        query = f"SELECT * FROM {table_name}"
        try:
            self.data = pd.read_sql(query, self.engine)
        except exc.SQLAlchemyError as e:
            raise e
        
    def get_data(self) -> pd.DataFrame:
        """Return the data"""
        if self.data is not None:
            return self.data
        else:
            raise ValueError("No data loaded")