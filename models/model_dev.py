import logging
from abc import ABC, abstractmethod
from prophet import Prophet

class Model(ABC):
    
    @abstractmethod
    def train(self,train):
        pass
    
    
class ProphetModel(Model):
        
    def train(self, train, **kwargs):
        try:
            m = Prophet( **kwargs)
            self.history = train
            m.fit(train)
            logging.info("Model training")
            return m
        except Exception as ex:
            logging.error("Error in cleaning data: {ex}")
            raise ex
        
    
            