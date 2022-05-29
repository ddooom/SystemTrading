import time
from datetime import datetime, timedelta
import pandas as pd
import pyupbit


def get_ohlcv(config):
    """ [업비트 코인의 OHLCV 받기]
    
    config의 dataset에 따른 특정 코인의 open, high, low, close, volume 받기
    
    Args:
        config (easydict.EasyDict) : config 
    
    Returns:
        pd.DataFrame : config의 ticker, interval, to, count에 따른 OHLCV 데이터프레임
    
    """
    
    if config.dataset.interval == 'minute60':
        date_diff = timedelta(hours=config.dataset.count)
    elif config.dataset.interval == 'day':
        date_diff = timedelta(days=config.dataset.count)
    elif config.dataset.interval == 'week':
        date_diff = timedelta(weeks=config.dataset.count)
    
    to_date = (datetime.strptime(config.dataset.to, '%Y%m%d %H')+timedelta(hours=1)).strftime("%Y%m%d %H")
    from_date = (datetime.strptime(to_date,'%Y%m%d %H')-date_diff).strftime("%Y%m%d %H")
    
    # ohlcv를 불러오는 url이 nonetype을 불러오는 경우를 방지
    for _ in range(10):
        df = pyupbit.get_ohlcv_from(ticker = config.dataset.ticker,
                                    interval = config.dataset.interval,
                                    fromDatetime = from_date,
                                    to = to_date,
                                    period = 0)
        if isinstance(df, pd.DataFrame):
            break
        time.sleep(0.1)
    
    df = df.iloc[-config.dataset.count:,:]
    
    return df

