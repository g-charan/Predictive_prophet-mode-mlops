import logging
import pandas as pd
from zenml import step
from models.model_dev import ProphetModel
from  .config import ModelNameConfig
from prophet import Prophet
from zenml.client import Client
import mlflow
import mlflow.prophet
from mlflow.models.signature import infer_signature

experiment_tracker  = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name,enable_cache=False)
def train_model(train:pd.DataFrame, config: str) -> Prophet:
    model = None
    if config == "Prophet":
        model = ProphetModel()
        trained_model = model.train(train)
        mlflow.prophet.log_model(
                pr_model=trained_model,
                artifact_path="model",
                signature=infer_signature(train),
                registered_model_name="MODEL_2"
            )
        return trained_model
    else:
        raise ValueError("Model not supported")