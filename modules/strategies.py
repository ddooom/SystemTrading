import copy
import pandas as pd

from modules import upbitfunc


def get_strategy(config):
    
    if (config.trade.strategy == 'Volatility Break-out') and (config.trade.cycle == 'day'):
            strategy = VolatilityBreakoutDayCycle(config)
            
    return strategy


class Strategy:
    def is_buy(self):
        None 
        

class VolatilityBreakoutDayCycle(Strategy):
    def __init__(self, config):
        self.config = config
    
    def is_buy(self):
        open_price = upbitfunc.get_ohlcv(self.config)['open'][-1]
        target_price = self.get_target_price()
        
        return open_price >= target_price
        
    def get_target_price(self):
        temp_config = copy.deepcopy(self.config)
        temp_config.dataset.interval = 'minute60'
        temp_config.dataset.count = 48
        hour_df = upbitfunc.get_ohlcv(temp_config)

        hour_df = hour_df.reset_index()
        open_idx, close_idx = hour_df[hour_df['index'].dt.hour == self.config.trade.open_hour].index[0:2]
        hour_df = hour_df.iloc[open_idx:close_idx].reset_index(drop=True)

        day_df = pd.DataFrame(index=range(1))
        day_df['ds'] = hour_df.iloc[0]['index']
        day_df['open'] = hour_df.iloc[0]['open']
        day_df['high'] = max(hour_df['high'])
        day_df['low'] = min(hour_df['low'])
        day_df['close'] = hour_df.iloc[-1]['close']
        day_df['volume'] = sum(hour_df['volume'])
        day_df['value'] = sum(hour_df['value'])

        target_price = day_df['close'][0] + (day_df['high'][0] - day_df['low'][0]) * self.config.trade.parameters.k

        return target_price
    
    def _is_train_cycle(self):
        temp_hour = int(self.config.dataset.to[-2:]) - self.config.trade.open_hour
        is_train = (temp_hour % self.config.model.cycle) == 0
        
        return is_train