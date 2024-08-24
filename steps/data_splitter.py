from zenml.logger import get_logger
from steps.src.model_building import DataSplitter 
logger = get_logger(__name__)

import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
from zenml import step

@step
def split_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,'X_train'],
    Annotated[pd.DataFrame,'X_test'],
    Annotated[pd.DataFrame,'y_train'],
    Annotated[pd.DataFrame,'y_test']
]:
    try:
        data_splitter = DataSplitter(df,features = df.drop('Unit Price',axis = 1).columns,target = 'Unit Price')
        X_train, X_test, y_train, y_test = data_splitter.split()
        logger.info("Data split completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Error while splitting data")
        raise e
        