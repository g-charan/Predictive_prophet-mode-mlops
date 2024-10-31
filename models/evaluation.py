import logging
from abc import ABC, abstractmethod

import pandas as pd
from prophet import Prophet
from statsmodels.tools.eval_measures import rmse


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, forecast: pd.DataFrame, test: pd.DataFrame) -> float:
        pass



# class R2Score(Evaluation):
#     """
#     Evaluation strategy that uses R2 Score
#     """
#     def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
#         """
#         Args:
#             y_true: np.ndarray
#             y_pred: np.ndarray
#         Returns:
#             r2_score: float
#         """
#         try:
#             logging.info("Entered the calculate_score method of the R2Score class")
#             r2 = r2_score(y_true, y_pred)
#             logging.info("The r2 score value is: " + str(r2))
#             return r2
#         except Exception as e:
#             logging.error(
#                 "Exception occurred in calculate_score method of the R2Score class. Exception message:  "
#                 + str(e)
#             )
#             raise e


class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_score(self, test: pd.DataFrame, forecast: pd.DataFrame) -> float:
        try:
            
            predictions = forecast.iloc[-30:]['yhat']
            rootse = rmse(predictions,test['y'])
            return rootse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the RMSE class. Exception message:  "
                + str(e)
            )
            raise e

