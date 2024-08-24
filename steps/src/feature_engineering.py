from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class FeatureEngineering(ABC):
    """Abstract class for feature"""
    
    @abstractmethod
    def fit_transform(self, df:pd.DataFrame , columns = List[str]) -> pd.DataFrame:
        """Fit and transform the data"""
        pass
    
    
    
class DataFeatureEngineer(FeatureEngineering):
    def __init__(self, date_format: str = "%m-%d-%Y"):
        self.date_format = date_format
        
    def fit_transform(self, df:pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for column in columns:
            df = self._split_date(df, column)
        return df
    
    def _split_date(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df[column] = pd.to_datetime(df[column], format=self.date_format)
        df[column + "_year"] = df[column].dt.year
        df[column + "_month"] = df[column].dt.month
        df[column + "_day"] = df[column].dt.day
        return df