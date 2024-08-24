from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing_extensions import Annotated
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLflowExperimentTracker
from zenml import step
from typing import Tuple, List
import pandas as pd
import mlflow
import mlflow.sklearn
from zenml.logger import get_logger

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(experiment_tracker, MLflowExperimentTracker):
    raise RuntimeError("This step requires an MLflowExperimentTracker to be set.")

@step(experiment_tracker = 'mlflow_tracker',
      settings = {"experiment_tracker.mlflow": {"experiment_name": "linear_regression"}},
      enable_cache=False,output_materializers = [SKLearnModelMaterializer,ListMaterializer])
def sklearn_train(
    X_train: Annotated[pd.DataFrame, 'X_train'],
    y_train: Annotated[pd.Series, 'y_train']
) -> Tuple[
    Annotated[LinearRegression, 'model'],
    Annotated[List[str], 'predictors']
]:
    try:
        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow.sklearn.autolog()
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            predictors = X_train.columns.tolist
            return model, predictors
    except Exception as e:
        logger.error(e)
        raise e