import time

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import pyupbit

from modules import utils, upbitfunc, models, strategies


def model_validation(config, date_list):
    """ [모델 성능 검증]
    
    config의 model을 사용하여 date_list의 날짜의 예측 값을 구하여 MAE로 성능을 검증
    
    Args:
        config (easydict.EasyDict) : config 
        date_list (list) : YYYYMMDD HH 형식의 날짜들로 이루어진 리스트
    
    Returns:
        (pd.DataFrame, float) : (print된 date별 y, yhat이 기록된 DataFrame, MAE)
    
    """
    
    result = pd.DataFrame(columns = ['start_train_date', 'end_train_date', 'predict_date', 'y', 'yhat', 'time'])
    
    print(f'Validate Model with {len(date_list)} data')
    print('┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓')
    print('┃  idx ┃             train date              ┃   predict date   ┃      y      ┃     yhat    ┃    time   ┃')
    print('┣━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━╋━━━━━━━━━━━━━╋━━━━━━━━━━━┫')
    for i, date in enumerate(date_list):
        start_time = time.time()
        config.dataset.to = date

        # define model
        model = models.get_model(config)

        # train model
        start_train_date = model._set_train_data()['ds'][0].strftime('%Y-%m-%d %H:00')
        end_train_date = model._set_train_data()['ds'][len(model._set_train_data())-1].strftime('%Y-%m-%d %H:00')
        model.train()

        # prediction
        pred = model.predict()
        pred_date = pred[0].strftime('%Y-%m-%d %H:00')
        yhat = pred[1]
        
        # get ground truth
        gt = pyupbit.get_ohlcv(config.dataset.ticker, 
                               config.dataset.interval, 
                               count=1, 
                               to=pred[0] + timedelta(hours=1), 
                               period=0)
        gt_date = gt.index[0].strftime('%Y-%m-%d %H:00')
        y = gt['open'].values[0]
        
        # logging
        process_time = time.time() - start_time
        print(f'┃ {i: >4} ┃ {start_train_date} ~ {end_train_date} ┃ ', end='')
        if pred_date == gt_date:
            print(f'{pred_date} ┃ {y: >11} ┃ {yhat: >11.1f} ┃ {process_time: >5.2f} sec ┃')
        else:
            raise Exception("Date Error!")
        row_list = [start_train_date, end_train_date, pred_date, y, yhat, process_time]
        row_data = pd.DataFrame([row_list], columns = result.columns)
        result = result.append(row_data, ignore_index=True)
            
    print('┗━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━┛')
    
    result['start_train_date'] = pd.to_datetime(result['start_train_date'], format = '%Y%m%d %H')
    result['end_train_date'] = pd.to_datetime(result['end_train_date'], format = '%Y%m%d %H')
    result['predict_date'] = pd.to_datetime(result['predict_date'], format = '%Y%m%d %H')
    mae = np.abs(np.subtract(result['y'].values, result['yhat'].values)).mean()
    print(f'* MAE : {mae}')
    
    return result, mae


def backtesting(config, money, date_list):
    """ [전략, 모델을 이용한 백테스팅]
    
    config의 model과 strategy 사용하여 date_list의 날짜의 money를 투자했을 때, 예상되는 손익을 계산 
    
    Args:
        config (easydict.EasyDict) : config 
        money (int) : 백테스팅할 자산
        date_list (list) : YYYYMMDD HH 형식의 날짜들로 이루어진 리스트
    
    Returns:
        (pd.DataFrame, float) : (print된 택테스팅 정보가 기록된 DataFrame, 수익률)
    
    """
    
    result = pd.DataFrame(columns = ['datetime', 'current_price', 'predict_price', 'sell', 'buy', 'money', 'coin', 'time'])
    coin = 0
    initial_money = money

    print(f'Back-Testing Model with {len(date_list)} data')
    print('┏━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓')
    print('┃  idx ┃     datetime     ┃ current price ┃ predict price ┃  sell ┃  buy  ┃   money  ┃    coin    ┃    time   ┃')
    print('┣━━━━━━╋━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━┫')

    for i, date in enumerate(date_list):
        start_time = time.time()
        config.dataset.to = date
        buy = False
        sell = False
        pred_price = 0
        current_price = upbitfunc.get_ohlcv(config)['open'][-1]

        # get model and strategy
        model = models.get_model(config)
        strategy = strategies.get_strategy(config)
        
        # sell
        if (int(date[-2:]) == config.trade.open_hour) and (coin > 0):
            money = current_price * coin * 0.9995
            coin = 0
            sell = True

        # satisfied strategy`s buying condition
        if strategy.is_buy() and (money > 0):
            
            # check cycle whether train model or not
            if config.dataset.interval == 'minute60':
                target_time = int(config.dataset.to[-2:])
                temp_time = target_time - config.trade.open_hour
                if temp_time <= 0: temp_time += 24
                is_train = (temp_time % config.model.cycle == 0)
            
            # train model or load model according to training cycle
            if is_train:
                model.train()
                model.save()
            else:
                model.load()
            
            # predict price
            pred = model.predict()
            pred_price = pred[1]
            
            # buy if prediction is higher than current price
            if current_price * config.trade.increase_rate <= pred_price:
                coin += money * 0.9995 / current_price
                money = 0
                buy = True

        # logging
        process_time = time.time() - start_time
        print(f'┃ {i: >4} ┃ {date[:4]}-{date[4:6]}-{date[6:8]} {date[-2:]}:00 ┃ {current_price: >13} ┃ {pred_price: >13.1f} ', end='')
        print(f'┃ {str(sell): >5} ┃ {str(buy): >5} ┃ {round(money): >8} ┃ {coin: >10.7f} ┃ {process_time: >5.2f} sec ┃')
        row_list = [date, current_price, pred_price, sell, buy, money, coin, process_time]
        row_data = pd.DataFrame([row_list], columns = result.columns)
        result = result.append(row_data, ignore_index=True)
        
    print('┗━━━━━━┻━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━┻━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━┛')
    
    result['datetime'] = pd.to_datetime(result['datetime'], format = '%Y%m%d %H')
    stock_yield = ((money/initial_money)-1)*100
    print(f'* Stock Yield : {stock_yield:.3f} %')
    
    return result, stock_yield