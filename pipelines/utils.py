import logging

import pandas as pd
from models.data_processing import DataCleaning, DataProcess


def get_data_for_test():
    try:
        df = pd.read_csv("./data/expense_data_1.csv",parse_dates=["DATE"],dayfirst=True)
        df = df.sample(n=100)
        preprocess_strategy = DataProcess()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        # df.drop(["review_score"], axis=1, inplace=True)
        # result = df.to_json(orient="split")
        return df
    except Exception as e:
        logging.error(e)
        raise e
