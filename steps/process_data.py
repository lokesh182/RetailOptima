import pandas as pd
from zenml import step
from zenml.logger import get_logger
from steps.src.data_process import CategoricalEncoder
from steps.src.feature_engineering import DateFeatureEngineer

logger = get_logger(__name__)


@step
def categorical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    try:
        encoder = CategoricalEncoder(method= "onehot",categories = "auto")
        df = encoder.fit_transform(df,columns = ["product_id","product_category_name"])
        logger.info("Categorical data encoded")
        return df
    except Exception as e:
        logger.error("Error while encoding categorical data")
        raise e

@step

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    try:
        date_engineer = DateFeatureEngineer(date_format = "%d-%m-%Y")
        df_transformed = date_engineer.fit_transform(df,columns = ["month_year"])
        logger.info("Feature engineering completed")
        
        df_transformed.drop(columns = ['id',"month_year"],inplace = True)
        return df_transformed
        
    except Exception as e:
        logger.error("Error while feature engineering")
        raise e
        
        
    