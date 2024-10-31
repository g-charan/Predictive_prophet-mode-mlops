import numpy as np
import json
import pandas as pd
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from pydantic import BaseModel
from steps.Ingest_data import ingest_df
from steps.data_cleaning import clean_df
from steps.model_train import train_model
from steps.evaluate import evaluate_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

from pipelines.utils import get_data_for_test


@step(enable_cache=False)
def dynamic_importer() -> pd.DataFrame:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Run an inference request against a prediction service"""

    # service.start(timeout=10)  # should be a NOP if already started
    # data = json.loads(data)
    # data.pop("columns")
    # data.pop("index")
    # columns_for_df = [
    #     "payment_sequential",
    #     "payment_installments",
    #     "payment_value",
    #     "price",
    #     "freight_value",
    #     "product_name_lenght",
    #     "product_description_lenght",
    #     "product_photos_qty",
    #     "product_weight_g",
    #     "product_length_cm",
    #     "product_height_cm",
    #     "product_width_cm",
    # ]
    # df = pd.DataFrame(data)
    # json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    # data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

class DeploymentTriggerConfig(BaseModel):
    min_rmse: float = 10.0
    
@step
def deployment_trigger(rmse: float, config: DeploymentTriggerConfig) -> bool:
    return rmse <= config.min_rmse

@pipeline(settings={"docker": docker_settings}
          ,enable_cache=False)
def continuous_deployment_pipeline(
    data_path: str,
    rmse: float = 10.0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    
):
    # Link all the steps artifacts together
    df = ingest_df(data_path)
    train,test = clean_df(df)
    model = train_model(train,"Prophet")
    rmse = evaluate_model(test,model)
    deployment_decision = deployment_trigger(
        rmse=rmse, 
        config=DeploymentTriggerConfig(min_rmse=10.0)  # Correctly passing config
    )
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=True,
        workers=workers,
        timeout=timeout,
    )
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
