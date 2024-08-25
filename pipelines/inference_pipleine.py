from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML

from steps.ingest_data import ingest_data
from steps.predict_step import predictor
from steps.prediction_service_loader import bentoml_prediction_service_loader
from steps.process_data import categorical_encoding,feature_engineering

docker_settings = DockerSettings(required_integrations=[BENTOML])  # Define the docker_settings variable

@pipeline(settings={"docker": docker_settings})
def inference_pipeline_retail(
    model_name: str, pipeline_name: str, step_name: str
):
    inference_data = ingest_data(table_name = "retail_prices",for_predict = True)
    df_processed = categorical_encoding(inference_data)
    df_transformed = feature_engineering(df_processed)
    prediction_service = bentoml_prediction_service_loader(
        model_name = model_name,pipeline_name = pipeline_name,step_name = step_name
    )
    predictor(inference_data = df_transformed,service = prediction_service)
  
    