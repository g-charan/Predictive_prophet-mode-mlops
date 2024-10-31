import logging
import pandas as pd
from zenml import step
from models.evaluation import RMSE
from typing import Tuple
from typing_extensions import Annotated
from prophet import Prophet

from zenml.client import Client
import mlflow
import mlflow.prophet
from mlflow.models import infer_signature
experiment_tracker  = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model( test: pd.DataFrame, model: Prophet) -> Annotated[float, "rmse"]:
        try:
            future = model.make_future_dataframe(periods=10)
            forecast = model.predict(future)
            # Using the RMSE class for root mean squared error calculation
            rmse_class = RMSE()
            rmse = rmse_class.calculate_score(test, forecast)
            
            mlflow.log_metric("rmse", rmse)
            print(mlflow.get_artifact_uri())
            print("RMSE score:", rmse)
            return rmse
        except Exception as ex:
            logging.error("Error in cleaning data: {ex}")
            raise ex