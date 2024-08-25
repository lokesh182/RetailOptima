import click

from constants import MODEL_NAME, PIPELINE_NAME, STEP_NAME
from pipelines.inference_pipleine import inference_pipeline_retail

from pipelines.training_pipeline import training_retail

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type = click.Choice([DEPLOY,PREDICT,DEPLOY_AND_PREDICT]),
    default = "deploy_and_predict",
    help = "Choose the configuration to run",
)

def main(config: str):
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    
    if deploy:
        training_retail()
        
    
    if predict:
        ans = inference_pipeline_retail(model_name = MODEL_NAME, pipeline_name = PIPELINE_NAME, step_name = STEP_NAME)
        print(ans)





    
if __name__ == "__main__":
    main()