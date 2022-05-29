import warnings
import pandas as pd

from fbprophet import Prophet

from modules import upbitfunc


def get_model(config):
    
    if config.model.name == 'prophet':
            model = ProphetModel(config)
            
    return model


class Model:
    def train(self):
        None
    
    def _set_train_data(self):
        None
    
    def predict(self):
        None
    
    def save(self):
        None
    
    def load(self):
        None

        
class ProphetModel(Model):
    def __init__(self, config):
        self.config = config
        self.model = Prophet(**self.config.model.parameters)
        
    def _set_train_data(self):
        df = upbitfunc.get_ohlcv(self.config)
        df = df.reset_index()
        df['ds'] = df['index']
        df['y'] = df['close']
        data = df[['ds','y']]
        
        return data
    
    def train(self):
        data = self._set_train_data()
        self.model.fit(data)
    
    def predict(self):
        future = self.model.make_future_dataframe(periods=24, freq='H')
        predictions = self.model.predict(future).iloc[-24:, :]
        prediction = predictions[predictions['ds'].dt.hour == self.config.trade.open_hour][['ds', 'yhat']].values[0]
        
        return prediction
