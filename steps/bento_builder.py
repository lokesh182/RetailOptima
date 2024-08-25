from constants import MODEL_NAME
from zenml import __version__ as zenml_version
from zenml.integrations.bentoml.steps import bento_builder_step


ben_to_builder = bento_builder_step.with_options(
    parameters = dict(
        model_name = MODEL_NAME,
        zenml_version = "sklearn",
        service="service.py:svc",
        labels = {"framework": "sklearn", "dataset": "retail", "zenml_version":"0.41.0",},
        exclude =["data"],
        python = {"packages":["zenml","scikit-learn"],
                  
                  
         },
    ),
    )