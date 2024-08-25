from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class CategoricalEncoder:
    """
    This class applies encoding to categorical variables. 
    
    Parameters
    ----------
    method: str, default="onehot"
        The method to encode the categorical variables. Can be "onehot" or "ordinal".
    
    categories: 'auto' or a list of lists, default='auto'
        Categories for the encoders. Must match the number of columns. If 'auto', categories are determined from data.
    """
    def __init__(self, method="onehot", categories='auto'):
        self.method = method
        self.categories = categories
        self.encoders = {}
        
    def fit(self, df, columns):
        """
        This function fits the encoding method to the provided data.
        
        Parameters
        ----------
        df: pandas DataFrame
            The input data to fit.
            
        columns: list of str
            The names of the columns to encode.
        """
        for col in columns:
            if self.method == "onehot":
                self.encoders[col] = OneHotEncoder(sparse_output=False, categories=self.categories)
            elif self.method == "ordinal":
                self.encoders[col] = OrdinalEncoder(categories=self.categories)
            else:
                raise ValueError(f"Invalid method: {self.method}")
            # Encoders expect 2D input data
            self.encoders[col].fit(df[[col]])


            
    def transform(self, df, columns):
        """
        This function applies the encoding to the provided data.
        
        Parameters
        ----------
        df: pandas DataFrame
            The input data to transform.
            
        columns: list of str
            The names of the columns to encode.
        """
        df_encoded = df.copy()
        for col in columns:
            # Encoders expect 2D input data
            transformed = self.encoders[col].transform(df[[col]])
            if self.method == "onehot":
                # OneHotEncoder returns a 2D array, we need to convert it to a DataFrame
                transformed = pd.DataFrame(transformed, columns=self.encoders[col].get_feature_names_out([col]))
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), transformed], axis=1)
            else:
                df_encoded[col] = transformed
        return df_encoded
    
    def fit_transform(self,df,columns):
        """Fit and transform the data"""
        self.fit(df,columns)
        return self.transform(df,columns)
    
    