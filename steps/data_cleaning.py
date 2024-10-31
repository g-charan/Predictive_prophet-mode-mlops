import logging
import pandas as pd
from zenml import step
from models.data_processing import DataCleaning,DataSplitting,DataProcess
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "train"],
    Annotated[pd.DataFrame, "test"],

]:
    try:
        process_strat = DataProcess()
        data_cleaning = DataCleaning(df, process_strat)
        processed_data = data_cleaning.handle_data()
        
        divide_strat = DataSplitting()
        data_cleaning = DataCleaning(processed_data, divide_strat)
        train,test = data_cleaning.handle_data()
        logging.info("Data Cleaning completed")
        return train,test
    except Exception as ex:
        logging.error("Error in cleaning data: {ex}")
        raise ex
        