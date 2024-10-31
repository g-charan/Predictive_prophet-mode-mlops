import logging
from abc import ABC,abstractmethod
from typing import Union
from typing import Tuple
from typing_extensions import Annotated
import numpy as np
import pandas as pd

class DataSegregation(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass
    
class DataProcess(DataSegregation):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(["Note","Account","Note.1","Subcategory","Date","Amount","Category","Currency","Income/Expense","INR","Account.1","total","S.no"],axis=1)
            data.columns = ['y', 'ds']
            data['ds'] = pd.to_datetime(data['ds'])
            return data
        except Exception as e:
            logging.error(e)
            raise e

class DataSplitting(DataSegregation):
    def handle_data(self, data: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame,"train"],Annotated[pd.DataFrame,"test"]]:
        train = data.iloc[:-30]
        test = data.iloc[-30:]
        return train,test
    
class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataSegregation) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)