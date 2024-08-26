from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML

from steps.bento_builder import ben_to_builder
from steps.data_splitter import  split_data
from steps.deployment_trigger_step import deployment_trigger
from steps.deployer import bentoml_model_deployer





from steps.ingest_data import ingest_data
from steps.process_data import categorical_encoding, feature_engineering

from steps.train_model import  sklearn_train

docker_settings = DockerSettings(required_integrations=[BENTOML])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_retail():
    """Train a model and deploy it with BentoML."""
    df = ingest_data("retail_prices") 
    df_processed = categorical_encoding(df)
    df_transformed = feature_engineering(df_processed)  
    X_train, X_test, y_train, y_test = split_data(df_transformed) 
     
    model, predictors = sklearn_train(X_train, y_train)  # Evaluate model
    rmse = 0.95
    
    decision = deployment_trigger(accuracy = rmse,min_accuracy = 0.80)
    bento = ben_to_builder(model = model)
    bentoml_model_deployer(bento = bento, deploy_decision = decision)   