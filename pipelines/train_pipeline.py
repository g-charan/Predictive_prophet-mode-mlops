from zenml import pipeline
from steps.Ingest_data import ingest_df
from steps.data_cleaning import clean_df
from steps.model_train import train_model
from steps.evaluate import evaluate_model
import logging


@pipeline(enable_cache=False)
def train_pipeline2(data_path: str):
    df = ingest_df(data_path)
    train,test = clean_df(df)
    model = train_model(train,"Prophet")
    rmse = evaluate_model(test,model)
    
    