import logging
import os
import pandas as pd
from zenml import step
from steps.src.data_loader import DataLoader

@step(enable_cache=False)

def ingest_data(
    table_name: str,
    for_predict: bool = False,
) -> pd.DataFrame:
    """Ingests data from a CSV file into a database table.
    
    Args:
        table_name: The name of the table to fill.
        
    Returns:
        The data from the CSV file.
    """
    try:
        data_loader = DataLoader( "postgresql://name:pass@127.0.0.1/CS001test")
        data_loader.load_data(table_name)
        df = data_loader.get_data()
        if for_predict:
            df= df.drop(columns = ["unit_price"])
            logging.info(f"Data loaded from table_name: {table_name}")
        return df
    except Exception as e:
        logging.error(f"Error while reading data from table_name: {table_name}")
        raise e