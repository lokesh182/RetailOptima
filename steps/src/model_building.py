from abc import ABC, abstractmethod
from typing import List, Tuple

import lxml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy import sqrt
from scipy.stats import shapiro
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

class DataSplitter:
    def __init__(self,df: pd.DataFrame,features: List[str], target: str, test_size: float = 0.2):
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size
        
    def split(self) -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        X = self.df[self.features]
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = self.test_size,shuffle=False)
        return X_train, X_test, y_train, y_test
    
class ModelRefinement:
    """Singleton class for refining a given model."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRefinement, cls).__new__(cls)
        return cls._instance

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.predictors = [x for x in self.model.model.exog_names if x != 'const']
        self.target = self.model.model.endog_names
        self.rmse = None

    def remove_insignificant_vars(self, alpha=0.05):
        """Remove insignificant variables based on p-value."""
        summary = self.model.summary().tables[1]
        summary_df = pd.DataFrame(summary.data)
        summary_df.columns = summary_df.iloc[0]
        summary_df = summary_df.drop(0)
        summary_df = summary_df.set_index(summary_df.columns[0])
        summary_df['P>|t|'] = summary_df['P>|t|'].astype(float)
        significant_vars = [var for var in self.predictors if summary_df.loc[var, 'P>|t|'] < alpha]
        self.predictors = significant_vars
        return significant_vars

    def check_multicollinearity(self):
        """Check multicollinearity among predictors."""
        exog = sm.add_constant(self.data[self.predictors])
        vif = pd.Series([variance_inflation_factor(exog.values, i) 
                         for i in range(exog.shape[1])], 
                        index=exog.columns)
        print("Variance Inflation Factors:")
        print(vif)

    def check_normality_of_residuals(self):
        """Check normality of residuals."""
        residuals = self.model.resid
        qqplot(residuals, line='s')
        plt.show()
        stat, p = shapiro(residuals)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')

    def check_homoscedasticity(self):
        """Check homoscedasticity."""
        residuals = self.model.resid
        plt.scatter(self.model.predict(), residuals)
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.axhline(y=0, color='red')
        plt.title('Residual vs. Predicted')
        plt.show()

    def validate(self, k=10):
        """Validate the model using K-Fold cross-validation."""
        kf = KFold(n_splits=k)
        y = self.data[self.target]
        X = sm.add_constant(self.data[self.predictors])
        errors = []
        
        for train, test in kf.split(X):
            model = sm.OLS(y.iloc[train], X.iloc[train]).fit()
            predictions = model.predict(X.iloc[test])
            mse = mean_squared_error(y.iloc[test], predictions)
            errors.append(mse) 
            print(f"MSE: {mse}")
        
        rmse = np.sqrt(np.mean(errors))
        self.rmse = rmse
        return rmse