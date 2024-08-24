from typing import List
from numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class CategoricalEncoder:
    """Class to encode categorical data"""
    
    def __init__(self, method: str = "onehot",categories: str = "auto"):
        self.method = method
        self.categories = categories
        self.encoder = None
        
    def fit(self, df: pd.DataFrame, columns) -> None:
        """Fit the encoder to the data""""
        for col in columns:
            if self.method == "onehot":
                self.encoder[col] = OneHotEncoder(sparse=False,categories=self.categories)
            elif self.method == "ordinal":
                self.encoder[col] = OrdinalEncoder(categories = self.categories)
            else:
                raise ValueError("Invalid method please use onehot or ordinal")
            
            self.encoder[col].fit(df[col])
            
    def transform(self,df,columns):
        """Transform the data"""
        df_encoded = df.copy()
        for col in columns:
            transformed = self.encoder[col].transform(df[col])
            if self.method == "onehot":
                transformed = pd.DataFrame(transformed, columns = self.encoder[col].get_feature_names_out())
                df_encoded = pd.concat([df_encoded.drop(columns = [col]),transformed],axis = 1)
            else:
                df_encoded[col] = transformed
        return df_encoded
    
    def fit_transform(self,df,columns):
        """Fit and transform the data"""
        self.fit(df,columns)
        return self.transform(df,columns)