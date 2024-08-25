from typing import Dict,List
import numpy as np
import pandas as pd
from rich import print as rich_print
from zenml import step
from zenml.integrations.bentoml.services import BentoMLDeploymentService
import numpy as np
import logging


@step
def predictor(
    inference_data: pd.DataFrame,
    service: BentoMLDeploymentService,
    file_path: str = "pred.txt"
    
)->np.ndarray:
    
    service.start(timeout = 10)
    
    inference_data = inference_data.to_numpy()
    prediction = service.predict("predict_ndarray", inference_data)
    rich_print(file_path, prediction)
    np.savetxt(file_path, prediction, fmt='%d')
    logging.info(f"Predictions saved to {file_path}")
    return prediction